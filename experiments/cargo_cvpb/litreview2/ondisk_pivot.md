Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef95a-b06c-7942-aa5c-9b3c9fb1733a
--------
user
一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。

**硬约束: 只能用 ON-DISK 数据(无下载, 没有视频/tracklet 数据)**: market1501 / MSMT17 / occluded_duke / occluded_reid / occluded_posetrack_reid(image-based) / CARGO(航拍-地面 image) / AG-ReID.v2(航拍-地面 image) / **RSTPReid(text-to-image 行人检索, 有文本标注)** + 强 Swin/SOLIDER backbone(occluded_duke 73 / market 94 / CARGO 67 都有强 ckpt)。

问: 用这些 ON-DISK 数据, **单个最可能成的 CCF-B 方法稿方向**是什么? 重点考虑团队没碰过的 **RSTPReid text-to-image** 这条新模态(team 有强 image backbone 可当 visual encoder), 也可考虑跨域(market→occluded_reid)/多数据集泛化/新协议/训练侧创新。要给: 触发观察(在哪能找到一个具体失败) + 重定义 + 机制 + **廉价(最好零训练 frozen-feature)kill-switch** + 撞车核查(text-image ReID 很卷, IRRA/APTM/RDE/CADA 等)。**诚实: 若 text-image 也卷死、其他也都死路, 直说该把 Hubness 写 analysis short 收尾、别再撞。** 联网, 务实中文, verdict + 信心 1-10。
codex
我会先核对你们现有实验叙事，再联网查 RSTPReid/text-image ReID 的近年主线和撞车风险。这个问题需要先确认“还能不能争一个方法贡献”，不能靠印象判断。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/results.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "pwd && rg --files experiments | sed -n '1,120p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/decisions.md: No such file or directory

 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
rg: experiments: IO error for operation on experiments: No such file or directory (os error 2)

exec
/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/innovation_brainstorm.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/paper_materials/story.md: No such file or directory

codex
当前工作目录下面没有 `experiments/` 子目录，说明这次上下文目录本身可能就是一个实验/文献子目录。我先定位仓库根目录和已有文档，避免把路径假设错。
exec
/bin/zsh -lc 'find .. -maxdepth 4 -name results.md -o -name decisions.md -o -name story.md -o -name innovation_brainstorm.md | sort' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
exec
/bin/zsh -lc 'git rev-parse --show-toplevel' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-j69CA8eW' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-Ga4aZ4gp' (errno=Operation not permitted)
/Users/abdslm/Desktop/SOLIDER-REID

exec
/bin/zsh -lc 'ls -la' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
total 296
drwxr-xr-x@ 27 abdslm  staff    864 Jun 24 19:18 .
drwxr-xr-x@ 73 abdslm  staff   2336 Jun 24 19:10 ..
-rw-r--r--@  1 abdslm  staff   7526 Jun 24 16:25 B_CONTAINMENT_DESIGN.md
-rw-r--r--@  1 abdslm  staff   4670 Jun 24 16:43 GOPL_KILLSWITCH_DESIGN.md
-rw-r--r--@  1 abdslm  staff   4052 Jun 24 17:19 HUBNESS_KILLSWITCH_DESIGN.md
-rw-r--r--@  1 abdslm  staff   4673 Jun 24 18:14 OSAC_KILLSWITCH_DESIGN.md
-rw-r--r--@  1 abdslm  staff   7368 Jun 24 14:20 SYNTHESIS_METHODOLOGY.md
-rw-r--r--@  1 abdslm  staff  16038 Jun 24 13:58 all_papers.txt
drwxr-xr-x@ 31 abdslm  staff    992 Jun 24 14:13 analyses
drwxr-xr-x@ 30 abdslm  staff    960 Jun 24 13:58 batches
-rw-r--r--@  1 abdslm  staff  15674 Jun 24 16:15 cvpb_containment_full.log
-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
drwxr-xr-x@  5 abdslm  staff    160 Jun 24 17:44 debate
-rw-r--r--@  1 abdslm  staff   3928 Jun 24 17:41 hubness_debate.sh
-rw-r--r--@  1 abdslm  staff   2385 Jun 24 13:58 launch_30deep.sh
-rw-r--r--@  1 abdslm  staff  14500 Jun 24 17:23 novelty_hubness.md
-rw-r--r--@  1 abdslm  staff  16010 Jun 24 18:19 novelty_osac.md
-rw-r--r--@  1 abdslm  staff   3568 Jun 24 19:19 ondisk_pivot.md
drwxr-xr-x@  7 abdslm  staff    224 Jun 24 16:36 pivot
drwxr-xr-x@  6 abdslm  staff    192 Jun 24 17:08 reassess
-rw-r--r--@  1 abdslm  staff   4431 Jun 24 17:06 reassess.sh
drwxr-xr-x@  5 abdslm  staff    160 Jun 24 18:04 reassess2
-rw-r--r--@  1 abdslm  staff   3059 Jun 24 18:03 reassess2.sh
drwxr-xr-x@ 30 abdslm  staff    960 Jun 24 13:59 reviews
-rw-r--r--@  1 abdslm  staff   4533 Jun 24 16:22 strategic_pivot.sh
drwxr-xr-x@  7 abdslm  staff    224 Jun 24 14:29 validate
-rw-r--r--@  1 abdslm  staff   4945 Jun 24 14:22 validate_candidates.sh

web search: 
web search: RSTPReid text-to-image person re-identification dataset IRRA APTM RDE CADA 2024 2025
web search: 
web search: "APTM" "text-to-image person" RSTPReid
web search: 
web search: "APTM" "RSTPReid" "GitHub"
web search: 
web search: "CADA" "text-image person re-identification"
web search: 
web search: https://arxiv.org/abs/2303.12501
web search: 
web search: 'RSTPReid' in https://arxiv.org/abs/2308.09911
web search: 
web search: https://arxiv.org/abs/2308.09911
web search: 
web search: "RDE" "RSTPReID" "57.55"
web search: 
web search: https://arxiv.org/pdf/2308.09911
web search: 
web search: 'RSTPReID dataset' in https://arxiv.org/pdf/2308.09911
web search: 
web search: https://arxiv.org/pdf/2308.09911
web search: 
web search: 'RSTPReid' in https://arxiv.org/pdf/2312.01745
web search: 
web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
web search: 
web search: "RSTPReID" "Rank-1" "2024"
web search: 
web search: "RSTP" "text-based person retrieval" 2024
web search: 
web search: site:arxiv.org "RSTPReid"
web search: 
web search: github IRRA text-to-image person retrieval
web search: 
web search: https://arxiv.org/abs/2506.11036
web search: 
web search: https://arxiv.org/abs/2506.11036
web search: 
web search: https://arxiv.org/pdf/2506.11036
web search: 
web search: 'RSTPReid' in https://arxiv.org/pdf/2506.11036
codex
联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。
exec
/bin/zsh -lc "sed -n '1,240p' ondisk_pivot.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef95a-b06c-7942-aa5c-9b3c9fb1733a
--------
user
一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。

**硬约束: 只能用 ON-DISK 数据(无下载, 没有视频/tracklet 数据)**: market1501 / MSMT17 / occluded_duke / occluded_reid / occluded_posetrack_reid(image-based) / CARGO(航拍-地面 image) / AG-ReID.v2(航拍-地面 image) / **RSTPReid(text-to-image 行人检索, 有文本标注)** + 强 Swin/SOLIDER backbone(occluded_duke 73 / market 94 / CARGO 67 都有强 ckpt)。

问: 用这些 ON-DISK 数据, **单个最可能成的 CCF-B 方法稿方向**是什么? 重点考虑团队没碰过的 **RSTPReid text-to-image** 这条新模态(team 有强 image backbone 可当 visual encoder), 也可考虑跨域(market→occluded_reid)/多数据集泛化/新协议/训练侧创新。要给: 触发观察(在哪能找到一个具体失败) + 重定义 + 机制 + **廉价(最好零训练 frozen-feature)kill-switch** + 撞车核查(text-image ReID 很卷, IRRA/APTM/RDE/CADA 等)。**诚实: 若 text-image 也卷死、其他也都死路, 直说该把 Hubness 写 analysis short 收尾、别再撞。** 联网, 务实中文, verdict + 信心 1-10。
codex
我会先核对你们现有实验叙事，再联网查 RSTPReid/text-image ReID 的近年主线和撞车风险。这个问题需要先确认“还能不能争一个方法贡献”，不能靠印象判断。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/results.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "pwd && rg --files experiments | sed -n '1,120p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/decisions.md: No such file or directory

 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
rg: experiments: IO error for operation on experiments: No such file or directory (os error 2)

exec
/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/innovation_brainstorm.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/paper_materials/story.md: No such file or directory

codex
当前工作目录下面没有 `experiments/` 子目录，说明这次上下文目录本身可能就是一个实验/文献子目录。我先定位仓库根目录和已有文档，避免把路径假设错。
exec
/bin/zsh -lc 'find .. -maxdepth 4 -name results.md -o -name decisions.md -o -name story.md -o -name innovation_brainstorm.md | sort' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
exec
/bin/zsh -lc 'git rev-parse --show-toplevel' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-j69CA8eW' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-Ga4aZ4gp' (errno=Operation not permitted)
/Users/abdslm/Desktop/SOLIDER-REID

exec
/bin/zsh -lc 'ls -la' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
total 296
drwxr-xr-x@ 27 abdslm  staff    864 Jun 24 19:18 .
drwxr-xr-x@ 73 abdslm  staff   2336 Jun 24 19:10 ..
-rw-r--r--@  1 abdslm  staff   7526 Jun 24 16:25 B_CONTAINMENT_DESIGN.md
-rw-r--r--@  1 abdslm  staff   4670 Jun 24 16:43 GOPL_KILLSWITCH_DESIGN.md
-rw-r--r--@  1 abdslm  staff   4052 Jun 24 17:19 HUBNESS_KILLSWITCH_DESIGN.md
-rw-r--r--@  1 abdslm  staff   4673 Jun 24 18:14 OSAC_KILLSWITCH_DESIGN.md
-rw-r--r--@  1 abdslm  staff   7368 Jun 24 14:20 SYNTHESIS_METHODOLOGY.md
-rw-r--r--@  1 abdslm  staff  16038 Jun 24 13:58 all_papers.txt
drwxr-xr-x@ 31 abdslm  staff    992 Jun 24 14:13 analyses
drwxr-xr-x@ 30 abdslm  staff    960 Jun 24 13:58 batches
-rw-r--r--@  1 abdslm  staff  15674 Jun 24 16:15 cvpb_containment_full.log
-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
drwxr-xr-x@  5 abdslm  staff    160 Jun 24 17:44 debate
-rw-r--r--@  1 abdslm  staff   3928 Jun 24 17:41 hubness_debate.sh
-rw-r--r--@  1 abdslm  staff   2385 Jun 24 13:58 launch_30deep.sh
-rw-r--r--@  1 abdslm  staff  14500 Jun 24 17:23 novelty_hubness.md
-rw-r--r--@  1 abdslm  staff  16010 Jun 24 18:19 novelty_osac.md
-rw-r--r--@  1 abdslm  staff   3568 Jun 24 19:19 ondisk_pivot.md
drwxr-xr-x@  7 abdslm  staff    224 Jun 24 16:36 pivot
drwxr-xr-x@  6 abdslm  staff    192 Jun 24 17:08 reassess
-rw-r--r--@  1 abdslm  staff   4431 Jun 24 17:06 reassess.sh
drwxr-xr-x@  5 abdslm  staff    160 Jun 24 18:04 reassess2
-rw-r--r--@  1 abdslm  staff   3059 Jun 24 18:03 reassess2.sh
drwxr-xr-x@ 30 abdslm  staff    960 Jun 24 13:59 reviews
-rw-r--r--@  1 abdslm  staff   4533 Jun 24 16:22 strategic_pivot.sh
drwxr-xr-x@  7 abdslm  staff    224 Jun 24 14:29 validate
-rw-r--r--@  1 abdslm  staff   4945 Jun 24 14:22 validate_candidates.sh

web search: 
web search: RSTPReid text-to-image person re-identification dataset IRRA APTM RDE CADA 2024 2025
web search: 
web search: "APTM" "text-to-image person" RSTPReid
web search: 
web search: "APTM" "RSTPReid" "GitHub"
web search: 
web search: "CADA" "text-image person re-identification"
web search: 
web search: https://arxiv.org/abs/2303.12501
web search: 
web search: 'RSTPReid' in https://arxiv.org/abs/2308.09911
web search: 
web search: https://arxiv.org/abs/2308.09911
web search: 
web search: "RDE" "RSTPReID" "57.55"
web search: 
web search: https://arxiv.org/pdf/2308.09911
web search: 
web search: 'RSTPReID dataset' in https://arxiv.org/pdf/2308.09911
web search: 
web search: https://arxiv.org/pdf/2308.09911
web search: 
web search: 'RSTPReid' in https://arxiv.org/pdf/2312.01745
web search: 
web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
web search: 
web search: "RSTPReID" "Rank-1" "2024"
web search: 
web search: "RSTP" "text-based person retrieval" 2024
web search: 
web search: site:arxiv.org "RSTPReid"
web search: 
web search: github IRRA text-to-image person retrieval
web search: 
web search: https://arxiv.org/abs/2506.11036
web search: 
web search: https://arxiv.org/abs/2506.11036
web search: 
web search: https://arxiv.org/pdf/2506.11036
web search: 
web search: 'RSTPReid' in https://arxiv.org/pdf/2506.11036
codex
联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。

exec
/bin/zsh -lc "sed -n '1,220p' novelty_hubness.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef8ee-34b4-7961-b665-4febe838df95
--------
user
我们要做 ReID 方法 **Gallery Hubness ReID**。重定义: 强 ReID 失败不是 pairwise 相似度病, 而是**少数 gallery 样本变成很多不同身份 query 的吸附点**(负向 in-degree / hub mass)=many-to-one 图库拓扑病。机制: 训练端 anti-hub margin(对全局误吸附 gallery 降权), memory bank 存 cross-ID top-k in-degree, negative 权重从'离anchor近'改'是否全局误吸附点', 测试仍单 embedding(不变 test-time trick)。

用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
codex
我会先按你给的六类逐项查英文论文和代码/项目页，优先拿论文标题、摘要和方法描述做证据，不把邻近概念自动算撞车。重点判断两个条件是否同时出现：把 gallery 负向 in-degree/hub mass 定义为失败主变量，以及训练端用它做 anti-hub 约束。
web search: 
web search: person re-identification hubness gallery hub in-degree negative top-k
web search: 
web search: Radovanovic hubness high dimensional retrieval hubs in space 2010
web search: 
web search: "Hubness-aware" loss metric learning
web search: 
web search: https://arxiv.org/abs/1911.10097
web search: 
web search: "hubness" "Market-1501"
web search: 
web search: "hubness" "vehicle re-identification"
web search: 
web search: person re-identification re-ranking k-reciprocal encoding CVPR 2017
web search: 
web search: batch hard triplet loss person re-identification Hermans Defense of the Triplet Loss
web search: 
web search: Proxy-NCA metric learning proxy anchor loss hard negative mining hubness
web search: 
web search: Multi-Similarity Loss deep metric learning hard mining global negative weighting
web search: 
web search: "Cross-domain Similarity Local Scaling" ReID person re-identification
web search: 
web search: "CSLS" "Market-1501"
web search: 
web search: Mutual Proximity hubness reduction high-dimensional data 2012
web search: 
web search: "Cross-domain similarity local scaling" "hubness" "word translation" "nearest neighbor"
web search: 
web search: https://arxiv.org/abs/1710.04087
web search: 
web search: "Mutual Proximity" "hubness" "high-dimensional" "nearest neighbor"
web search: 
web search: "Mutual Proximity" "Schnitzer" "Flexer" "high-dimensional"
web search: 
web search: "Local and global scaling reduce hubs"
web search: 
web search: "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data"
web search: 
web search: site:jmlr.org "Hubs in space"
web search: 
web search: https://www.jmlr.org/papers/v11/radovanovic10a.html
web search: 
web search: https://jmlr.org/papers/v13/schnitzer12a.html
web search: 
web search: https://arxiv.org/abs/1701.08398
web search: 
web search: https://arxiv.org/abs/1711.10378
web search: 
web search: "Pose2ID" "person re-identification"
web search: 
web search: https://arxiv.org/abs/2503.00938
web search: 
web search: https://arxiv.org/abs/2503.00938
web search: 
web search: https://arxiv.org/abs/1807.09975
web search: 
web search: "popularity bias" "image retrieval" "long-tail" "hubness"
web search: 
web search: "popularity bias" "information retrieval" recommender systems survey
web search: 
web search: "False Match Rate" "person re-identification"
web search: 
web search: "open-set" "person re-identification" "false positive"
web search: 
web search: "open set person re-identification"
web search: 
web search: https://arxiv.org/abs/1807.10482
web search: 
web search: https://arxiv.org/abs/2503.10526
web search: 
web search: https://arxiv.org/abs/1703.07737
web search: 
web search: "Gallery Hubness" ReID
codex
结论先给：**Gallery Hubness ReID 有条件存活，不是直接撞车；但“hubness + memory bank + 训练端重加权”在跨模态检索里已经很接近，必须正面引用 HAL / NeighborRetr，不能把 hubness-aware training 当成完全新概念。**  
我检索到的最强杀伤不是 ReID，而是 cross-modal retrieval。

| 类别 | verdict | 撞车判断 | 证据 | 信心 |
|---|---:|---|---|---:|
| (a) 高维检索 hubness / CSLS / MP / local scaling | **存活，但术语不新** | Radovanovic 已把 hubness 定义成 kNN 图里的高 `k-occurrence`，本质就是 in-degree；Schnitzer 的 local/global scaling、CSLS 都是 post-hoc 降 hub。**未见用于 person ReID 训练端 anti-hub**。 | Radovanovic JMLR 定义点在他人 kNN 中出现次数并形成 popular NN；Schnitzer JMLR 讨论对称化 NN 关系减 hub；CSLS 明确为缓解 hubness 的相似度修正。链接：[Radovanovic 2010](https://www.jmlr.org/papers/v11/radovanovic10a.html), [Schnitzer 2012](https://jmlr.org/papers/v13/schnitzer12a.html), [CSLS/Conneau 2017](https://arxiv.org/abs/1710.04087) | 9 |
| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
| (e) graph / relation ReID, SGGNN 等 | **存活，但叙事会被问到** | SGGNN 已经说 pairwise 独立相似度不够，要用 gallery-gallery / probe-gallery relation，并把图传播纳入训练和测试。区别是它预测 relation score，测试也走图；不是统计 gallery 负 in-degree，也不是单 embedding 输出。 | SGGNN 构造 probe-gallery pair 节点图，用 gallery-gallery similarity 做 message passing。链接：[SGGNN](https://arxiv.org/abs/1807.09975) | 8 |
| (f) long-tail / popularity-bias / false-match-rate retrieval | **存活，弱邻近** | 推荐系统 popularity bias 和 open-world ReID 的 imposter / non-target false match 都是邻近概念，但没有把 closed-set ReID 的 gallery 样本当“被很多不同 ID query 误吸附的 hub”来训练惩罚。 | Open-world ReID/APN 关注非目标人像 target-like imposters；long-tail person search 关注身份分布不均。链接：[Adversarial Open-World ReID](https://arxiv.org/abs/1807.10482), [Long-tail Person Search](https://arxiv.org/abs/2112.00527) | 7 |

**务实判断**

- **不直接撞 ReID**：我没检到 “Gallery Hubness ReID / gallery negative in-degree / anti-hub margin for person re-identification” 这类确切先例。
- **外域强撞 HAL / NeighborRetr**：如果我们只写“发现 hubness，用 memory bank 统计 hub，训练时给 hub negative 加权”，会被认为是把 cross-modal retrieval 的 hubness-aware training 移植到 ReID，创新杀伤大。
- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
tokens used
117,784
结论先给：**Gallery Hubness ReID 有条件存活，不是直接撞车；但“hubness + memory bank + 训练端重加权”在跨模态检索里已经很接近，必须正面引用 HAL / NeighborRetr，不能把 hubness-aware training 当成完全新概念。**  
我检索到的最强杀伤不是 ReID，而是 cross-modal retrieval。

| 类别 | verdict | 撞车判断 | 证据 | 信心 |
|---|---:|---|---|---:|
| (a) 高维检索 hubness / CSLS / MP / local scaling | **存活，但术语不新** | Radovanovic 已把 hubness 定义成 kNN 图里的高 `k-occurrence`，本质就是 in-degree；Schnitzer 的 local/global scaling、CSLS 都是 post-hoc 降 hub。**未见用于 person ReID 训练端 anti-hub**。 | Radovanovic JMLR 定义点在他人 kNN 中出现次数并形成 popular NN；Schnitzer JMLR 讨论对称化 NN 关系减 hub；CSLS 明确为缓解 hubness 的相似度修正。链接：[Radovanovic 2010](https://www.jmlr.org/papers/v11/radovanovic10a.html), [Schnitzer 2012](https://jmlr.org/papers/v13/schnitzer12a.html), [CSLS/Conneau 2017](https://arxiv.org/abs/1710.04087) | 9 |
| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
| (e) graph / relation ReID, SGGNN 等 | **存活，但叙事会被问到** | SGGNN 已经说 pairwise 独立相似度不够，要用 gallery-gallery / probe-gallery relation，并把图传播纳入训练和测试。区别是它预测 relation score，测试也走图；不是统计 gallery 负 in-degree，也不是单 embedding 输出。 | SGGNN 构造 probe-gallery pair 节点图，用 gallery-gallery similarity 做 message passing。链接：[SGGNN](https://arxiv.org/abs/1807.09975) | 8 |
| (f) long-tail / popularity-bias / false-match-rate retrieval | **存活，弱邻近** | 推荐系统 popularity bias 和 open-world ReID 的 imposter / non-target false match 都是邻近概念，但没有把 closed-set ReID 的 gallery 样本当“被很多不同 ID query 误吸附的 hub”来训练惩罚。 | Open-world ReID/APN 关注非目标人像 target-like imposters；long-tail person search 关注身份分布不均。链接：[Adversarial Open-World ReID](https://arxiv.org/abs/1807.10482), [Long-tail Person Search](https://arxiv.org/abs/2112.00527) | 7 |

**务实判断**

- **不直接撞 ReID**：我没检到 “Gallery Hubness ReID / gallery negative in-degree / anti-hub margin for person re-identification” 这类确切先例。
- **外域强撞 HAL / NeighborRetr**：如果我们只写“发现 hubness，用 memory bank 统计 hub，训练时给 hub negative 加权”，会被认为是把 cross-modal retrieval 的 hubness-aware training 移植到 ReID，创新杀伤大。
- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。

exec
/bin/zsh -lc "sed -n '1,220p' novelty_osac.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef920-53d9-7120-a67c-fc5bba459cd9
--------
user
我们要做 ReID 方法 **OSAC(Open-Set Spectral Over-Collapse)**。重定义: open-set ReID 的隐藏变量是 **seen-ID neural collapse 过强**——训练末期特征过度对齐 seen-ID prototype 几何, unseen-ID 可迁移身份证据低秩化/各向异性化挤到低能谱尾部; k-reciprocal 只能重排恢复不了被压没的维度。hubness 是症状非根因。机制: 训练期 spectral anti-collapse(spectral floor + Top-PC Dropout)保 unseen 证据, 测试仍单 embedding。

用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'seen-ID 过坍缩定义成 open-set ReID 失败变量 + 训练期抗坍缩保 unseen 迁移': (a)Neural Collapse(Papyan 2020)用到 ReID / open-set / metric learning 没; (b)dimensional collapse / rank collapse in SSL/contrastive(Jing et al, Hua et al); (c)VICReg/Barlow Twins/whitening/decorrelation 泛化零件; (d)ProNet / prototype-projection / classifier-reuse ReID; (e)generalizable/open-set/DG ReID 里有没有'过坍缩伤 unseen 迁移'诊断; (f)feature whitening / group whitening ReID(AMS / SNR / IN 类); (g)spectral/rank regularization metric learning; (h)open-set recognition 里的 feature collapse。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
codex
我会按你列的八类逐一查最近和经典先例，重点不是泛泛找“谱/白化/collapse”，而是确认有没有把它精确定义为 open-set ReID 的 seen-ID 过坍缩失效变量，并且训练期用谱抗坍缩保护 unseen 证据。
web search: 
web search: Neural Collapse person re-identification open-set metric learning prototype ReID
web search: 
web search: "dimensional collapse" contrastive self-supervised learning Jing Hua
web search: 
web search: "ProNet" person re-identification prototype projection classifier reuse ReID
web search: 
web search: "over-collapse" "person re-identification"
web search: 
web search: "domain generalization" "person re-identification" "rank" "feature"
web search: 
web search: "overfitting to source identities" "person re-identification" "unseen"
web search: 
web search: person re-identification feature whitening group whitening AMS SNR IN
web search: 
web search: "Style Normalization and Restitution" person re-identification SNR
web search: 
web search: "spectral regularization" "deep metric learning" rank collapse
web search: 
web search: "spectrum" "deep metric learning" embedding regularization
web search: 
web search: "dimensional collapse" "deep metric learning"
web search: 
web search: "rank regularization" "embedding" "metric learning"
web search: 
web search: "feature collapse" "open set recognition"
web search: 
web search: "Open-Set Recognition" "neural collapse"
web search: 
web search: site:openaccess.thecvf.com "Neural Collapse" "Open Set Recognition"
web search: 
web search: "Neural Collapse" "Deep Metric Learning"
web search: 
web search: "Neural Collapse" "supervised contrastive"
web search: 
web search: "Neural Collapse" transfer learning generalization unseen classes
web search: 
web search: "neural collapse" "hurts" transfer learning unseen classes
web search: 
web search: Papyan Han Donoho Prevalence of Neural Collapse during terminal phase of deep learning PNAS 2020
web search: 
web search: "Neural Collapse" "re-identification"
web search: 
web search: neural collapse reidentification ReID person ID classifier weights prototypes
web search: 
web search: "Top-PC Dropout"
web search: 
web search: "dropout" "top principal components" representations
web search: 
web search: "batch spectral shrinkage" transfer learning singular values representation
web search: 
web search: "Batch Spectral Shrinkage"
web search: 
web search: "Batch Spectral Shrinkage" arxiv
web search: 
web search: "Transferability vs. Discriminability" "Deep Transfer Learning" "Batch Spectral Penalization"
web search: 
web search: "generalizable person re-identification" "overfitting" "source" "style"
web search: 
web search: "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" arxiv
web search: 
web search: "feature decorrelation" "person re-identification"
web search: 
web search: "spectrum" "person re-identification" "feature" "regularization"
web search: 
web search: "spectral floor" deep learning representation
web search: 
web search: "anisotropy" "person re-identification" feature embedding
web search: 
web search: "hubness" "person re-identification"
web search: 
web search: "classifier weights" "person re-identification" "prototype"
web search: 
web search: https://arxiv.org/abs/2008.08186
web search: 
web search: https://arxiv.org/abs/2005.11037
web search: 
web search: https://arxiv.org/abs/2112.15121
web search: 
web search: 'training longer leads to worse last-layer features' in https://arxiv.org/abs/2202.08384
codex
截至 2026-06-24 检索后的判断：**OSAC 主切口存活，但机制零件撞车很多**。不能把“variance/covariance/whitening/谱正则防 collapse”写成主创新；主创新必须写成 **open-set ReID 中 seen-ID prototype 几何过强导致 unseen 身份证据谱尾化，并用训练期抗坍缩保护单 embedding 的可迁移维度**。

| 类别 | Verdict | 撞车点 / 未撞切口 | 信心 |
|---|---|---|---|
| (a) Neural Collapse 到 ReID/open-set/metric learning | **存活，强相邻撞车** | Papyan NC 已定义 train class feature 向 class mean / ETF / classifier self-dual collapse。NC 已用于 transfer/OOD/OSR 解释或检测，且 Galanti 甚至说 NC 可泛化到新类；Hui 则指出 NC 可能让 downstream feature 变差。**没查到把 NC 明确落到 person ReID seen-ID 过坍缩作为 open-set ReID 失败变量**。证据：[Papyan 2020](https://arxiv.org/abs/2008.08186)、[Hui 2022](https://arxiv.org/abs/2202.08384)、[Galanti 2021](https://arxiv.org/abs/2112.15121)、[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823) | 8/10 |
| (b) SSL/contrastive dimensional collapse | **机制撞车，问题存活** | Jing/Hua 已经把“维度坍缩、强相关、低维子空间”讲得很清楚。OSAC 的“低秩化/各向异性化”概念不能当新现象，只能当迁移到 ReID 的诊断变量。证据：[Jing et al.](https://arxiv.org/abs/2110.09348)、[Hua et al.](https://arxiv.org/abs/2105.00470) | 9/10 |
| (c) VICReg / Barlow / whitening / decorrelation | **零件严重撞车** | spectral floor 类似 VICReg variance floor；去相关/白化类似 Barlow Twins、Whitening SSL。不能声称“防 collapse 的谱地板/去相关”新。切口在 supervised open-set ReID 的 seen-ID prototype over-collapse，而非 SSL collapse。证据：[VICReg](https://arxiv.org/abs/2105.04906)、[Barlow Twins](https://arxiv.org/abs/2103.03230)、[Whitening SSL](https://arxiv.org/abs/2007.06346) | 10/10 |
| (d) ProNet / prototype projection / classifier reuse ReID | **强相邻撞车，有杀伤** | ProNet 直接把 ReID classifier 权重当 seen-ID prototypes，并在推理时保留 prototype similarity 做检索特征。这会挑战“seen prototype 对齐一定伤 unseen”。但它没有诊断过坍缩，也不是训练期谱抗坍缩，且测试不是“仍单原始 embedding”。必须做 ProNet/ProNet-like 对照。证据：[ProNet](https://arxiv.org/abs/2308.10717) | 9/10 |
| (e) open-set / DG ReID 是否已有“过坍缩伤 unseen” | **核心切口存活** | Open-set ReID/GOM 已强调 query ID 可能不在 gallery、ReID 是 retrieval+verification；DG ReID 多数讲 domain/style overfit，不是 seen-ID neural collapse。没看到“训练末期 seen-ID prototype collapse 压掉 unseen identity evidence”的诊断。证据：[GOM](https://arxiv.org/abs/2011.11506)、[SNR](https://arxiv.org/abs/2005.11037)、[MetaBIN](https://arxiv.org/abs/2011.14670)、[Style Interleaved](https://arxiv.org/abs/2207.03132) | 8/10 |
| (f) feature whitening / GW / IN 类 ReID | **中等撞车** | AMS 已用 IN + Group Whitening 做 DG ReID；SNR 明确说 IN 会丢 identity-relevant 信息再 restitution；CC-ReID 也有 feature decorrelation。OSAC 要避开“又一个 whitening ReID”叙事，强调 classifier-prototype over-collapse 与谱尾 unseen 证据。证据：[AMS](https://arxiv.org/abs/2210.10409)、[SNR](https://arxiv.org/abs/2005.11037)、[CC-ReID decorrelation](https://arxiv.org/abs/2410.05536)、[RaMoE decorrelation](https://arxiv.org/abs/2105.09156) | 9/10 |
| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
| (h) open-set recognition 里的 feature collapse | **相邻撞车，不杀** | OSR/OOD 已把 NC/known-class simplex geometry 用来做 unknown rejection，甚至 Deep Simplex classifier 系列利用 NC。多数是“利用 known collapse 区分 unknown”，不是“known collapse 过强会损害 unseen ReID retrieval”。证据：[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823)、[DMDSC](https://arxiv.org/abs/2605.00675) | 8/10 |

**总 verdict：存活，信心 7/10。**  
杀伤最大的不是某篇完全同题论文，而是三类 reviewer attack：VICReg/Barlow 说机制不新，AMS/SNR/feature decorrelation ReID 说 ReID 已做白化去相关，ProNet 说 seen prototypes 对 unseen retrieval 也可能有用。

OSAC 要站住，实验必须证明三件事：  
1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。
tokens used
83,709
截至 2026-06-24 检索后的判断：**OSAC 主切口存活，但机制零件撞车很多**。不能把“variance/covariance/whitening/谱正则防 collapse”写成主创新；主创新必须写成 **open-set ReID 中 seen-ID prototype 几何过强导致 unseen 身份证据谱尾化，并用训练期抗坍缩保护单 embedding 的可迁移维度**。

| 类别 | Verdict | 撞车点 / 未撞切口 | 信心 |
|---|---|---|---|
| (a) Neural Collapse 到 ReID/open-set/metric learning | **存活，强相邻撞车** | Papyan NC 已定义 train class feature 向 class mean / ETF / classifier self-dual collapse。NC 已用于 transfer/OOD/OSR 解释或检测，且 Galanti 甚至说 NC 可泛化到新类；Hui 则指出 NC 可能让 downstream feature 变差。**没查到把 NC 明确落到 person ReID seen-ID 过坍缩作为 open-set ReID 失败变量**。证据：[Papyan 2020](https://arxiv.org/abs/2008.08186)、[Hui 2022](https://arxiv.org/abs/2202.08384)、[Galanti 2021](https://arxiv.org/abs/2112.15121)、[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823) | 8/10 |
| (b) SSL/contrastive dimensional collapse | **机制撞车，问题存活** | Jing/Hua 已经把“维度坍缩、强相关、低维子空间”讲得很清楚。OSAC 的“低秩化/各向异性化”概念不能当新现象，只能当迁移到 ReID 的诊断变量。证据：[Jing et al.](https://arxiv.org/abs/2110.09348)、[Hua et al.](https://arxiv.org/abs/2105.00470) | 9/10 |
| (c) VICReg / Barlow / whitening / decorrelation | **零件严重撞车** | spectral floor 类似 VICReg variance floor；去相关/白化类似 Barlow Twins、Whitening SSL。不能声称“防 collapse 的谱地板/去相关”新。切口在 supervised open-set ReID 的 seen-ID prototype over-collapse，而非 SSL collapse。证据：[VICReg](https://arxiv.org/abs/2105.04906)、[Barlow Twins](https://arxiv.org/abs/2103.03230)、[Whitening SSL](https://arxiv.org/abs/2007.06346) | 10/10 |
| (d) ProNet / prototype projection / classifier reuse ReID | **强相邻撞车，有杀伤** | ProNet 直接把 ReID classifier 权重当 seen-ID prototypes，并在推理时保留 prototype similarity 做检索特征。这会挑战“seen prototype 对齐一定伤 unseen”。但它没有诊断过坍缩，也不是训练期谱抗坍缩，且测试不是“仍单原始 embedding”。必须做 ProNet/ProNet-like 对照。证据：[ProNet](https://arxiv.org/abs/2308.10717) | 9/10 |
| (e) open-set / DG ReID 是否已有“过坍缩伤 unseen” | **核心切口存活** | Open-set ReID/GOM 已强调 query ID 可能不在 gallery、ReID 是 retrieval+verification；DG ReID 多数讲 domain/style overfit，不是 seen-ID neural collapse。没看到“训练末期 seen-ID prototype collapse 压掉 unseen identity evidence”的诊断。证据：[GOM](https://arxiv.org/abs/2011.11506)、[SNR](https://arxiv.org/abs/2005.11037)、[MetaBIN](https://arxiv.org/abs/2011.14670)、[Style Interleaved](https://arxiv.org/abs/2207.03132) | 8/10 |
| (f) feature whitening / GW / IN 类 ReID | **中等撞车** | AMS 已用 IN + Group Whitening 做 DG ReID；SNR 明确说 IN 会丢 identity-relevant 信息再 restitution；CC-ReID 也有 feature decorrelation。OSAC 要避开“又一个 whitening ReID”叙事，强调 classifier-prototype over-collapse 与谱尾 unseen 证据。证据：[AMS](https://arxiv.org/abs/2210.10409)、[SNR](https://arxiv.org/abs/2005.11037)、[CC-ReID decorrelation](https://arxiv.org/abs/2410.05536)、[RaMoE decorrelation](https://arxiv.org/abs/2105.09156) | 9/10 |
| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
| (h) open-set recognition 里的 feature collapse | **相邻撞车，不杀** | OSR/OOD 已把 NC/known-class simplex geometry 用来做 unknown rejection，甚至 Deep Simplex classifier 系列利用 NC。多数是“利用 known collapse 区分 unknown”，不是“known collapse 过强会损害 unseen ReID retrieval”。证据：[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823)、[DMDSC](https://arxiv.org/abs/2605.00675) | 8/10 |

**总 verdict：存活，信心 7/10。**  
杀伤最大的不是某篇完全同题论文，而是三类 reviewer attack：VICReg/Barlow 说机制不新，AMS/SNR/feature decorrelation ReID 说 ReID 已做白化去相关，ProNet 说 seen prototypes 对 unseen retrieval 也可能有用。

OSAC 要站住，实验必须证明三件事：  
1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。

exec
/bin/zsh -lc "sed -n '1,180p' all_papers.txt" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
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
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Multi Queue for Unsupervised Person Re-identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
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
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf

codex
本地记录确认了你说的四条 frozen-image 路线已经被更便宜的工具/反证压住；其中 hubness 现在更适合保留为诊断素材。RSTPReid 方向如果要成，不能做“又一个 alignment 模块”，只能做能用零训练先杀的训练侧问题重定义。
exec
/bin/zsh -lc "find analyses reviews pivot reassess reassess2 debate validate -maxdepth 2 -type f | sort | sed -n '1,160p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
analyses/ALL_ANALYSES.txt
analyses/deep_1.txt
analyses/deep_10.txt
analyses/deep_11.txt
analyses/deep_12.txt
analyses/deep_13.txt
analyses/deep_14.txt
analyses/deep_15.txt
analyses/deep_16.txt
analyses/deep_17.txt
analyses/deep_18.txt
analyses/deep_19.txt
analyses/deep_2.txt
analyses/deep_20.txt
analyses/deep_21.txt
analyses/deep_22.txt
analyses/deep_23.txt
analyses/deep_24.txt
analyses/deep_25.txt
analyses/deep_26.txt
analyses/deep_27.txt
analyses/deep_28.txt
analyses/deep_3.txt
analyses/deep_4.txt
analyses/deep_5.txt
analyses/deep_6.txt
analyses/deep_7.txt
analyses/deep_8.txt
analyses/deep_9.txt
debate/clean/d_1.txt
debate/clean/d_2.txt
debate/d_1.md
debate/d_2.md
pivot/clean/novelty_gopl.txt
pivot/clean/novelty_hubness.txt
pivot/clean/novelty_osac.txt
pivot/clean/p_1.txt
pivot/clean/p_2.txt
pivot/clean/p_3.txt
pivot/novelty_gopl.md
pivot/p_1.md
pivot/p_2.md
pivot/p_3.md
reassess/clean/r_1.txt
reassess/clean/r_2.txt
reassess/clean/r_3.txt
reassess/r_1.md
reassess/r_2.md
reassess/r_3.md
reassess2/clean/x_1.txt
reassess2/clean/x_2.txt
reassess2/x_1.md
reassess2/x_2.md
reviews/deep_1.md
reviews/deep_10.md
reviews/deep_11.md
reviews/deep_12.md
reviews/deep_13.md
reviews/deep_14.md
reviews/deep_15.md
reviews/deep_16.md
reviews/deep_17.md
reviews/deep_18.md
reviews/deep_19.md
reviews/deep_2.md
reviews/deep_20.md
reviews/deep_21.md
reviews/deep_22.md
reviews/deep_23.md
reviews/deep_24.md
reviews/deep_25.md
reviews/deep_26.md
reviews/deep_27.md
reviews/deep_28.md
reviews/deep_3.md
reviews/deep_4.md
reviews/deep_5.md
reviews/deep_6.md
reviews/deep_7.md
reviews/deep_8.md
reviews/deep_9.md
validate/clean/v_1.txt
validate/clean/v_2.txt
validate/clean/v_3.txt
validate/clean/v_4.txt
validate/v_1.md
validate/v_2.md
validate/v_3.md
validate/v_4.md

exec
/bin/zsh -lc 'rg -n "RSTP|text|图文|文本|IRRA|RDE|APTM|CADA|hubness|kill|verdict|结论" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
./validate_candidates.sh:13:[B 主推] **非对称信息包含**: 航拍低清俯视=信息欠定(看不清脸/纹理), 地面高清正面=信息确定; 但所有 cross-view ReID 用**对称** cosine/MaxSim 匹配。重定义='跨视角不是对称对齐, 而是航拍身份证据=一个范围(宽分布)应被地面的窄分布**非对称包含**', 方差=该视角信息欠定度。机制: 每图建分布(均值+方差), 包含损失让航拍分布⊆地面分布。kill-switch(零训练): 冻结 Swin 提特征, 对称cosine vs 非对称包含距离(航拍宽高斯/地面窄高斯, 马氏或KL包含) on CARGO A↔G。
./validate_candidates.sh:18:ROLES[1]="你的角色=**蓝队杀手**。用联网搜索**尽全力杀死候选 B(非对称包含 for 跨视角)**: 找出让它不新颖的确切先例——重点查 PDA/Probabilistic Distribution Alignment(文本-图像包含)、probabilistic/Gaussian embedding ReID、OT-based ReID(CM-EMD/G2DA/CVFT)、set-based/distribution matching ReID、aerial-ground(VDT/GSAlign/AG-VPReID/SDPL)、cross-resolution ReID。逐个判断它们是否已经做了'非对称分布包含'或'低清query信息欠定建模'。如果杀不死, 明确说它靠什么活下来(切开点)。务实中文, 给 verdict(撞车/存活)+ 证据链接。"
./validate_candidates.sh:19:ROLES[2]="你的角色=**红队辩护**。为候选 B 辩护它是最强 B 类赌注: 用联网把它和 PDA/OT-ReID/probabilistic-embedding 的**切开点**打磨锋利(我们是 cross-VIEW 不是 text-image; 非对称方向航拍⊆地面由**成像物理**定不是纯视觉cost)。设计**最能打动 reviewer 的杀手级证据**(哪个消融/可视化/诊断指标一旦失败就推翻叙事)。再判断这个 idea 够不够一篇 B 类的体量。务实中文, 给信心分 1-10 + 证据设计。"
./validate_candidates.sh:20:ROLES[3]="你的角色=**独立裁判**。不预设立场, 用联网核查后**独立给 B/C/D 排序**, 并且——关键——基于我们那个 avg>MaxSim 观察, 看看有没有**比 B/C/D 更强的全新 re-frame**(用我们总结的 22 个重定义动作: 数学化/可测中间变量/对齐伤判别/因果/表示形态/改信号角色/顺序错了 等)。给出你认为最该做的 1 个, 和它的廉价零训练 kill-switch。务实中文。"
./validate_candidates.sh:21:ROLES[4]="你的角色=**kill-switch 批判员**。只盯一件事: 候选 B/C/D 的'零训练冻结Swin' kill-switch **是否真能证伪 re-framing**? 比如候选 B 的'非对称包含距离 beat 对称cosine'——会不会赢是因为别的混杂原因(比如只是 query/gallery 归一化差异、或方差只是反映难度而非信息欠定)? 怎么设计**对照**才能干净隔离'非对称包含'这个机制本身的功劳(参考方法论里的'替换机制破坏性对照')? 给出加固后的 kill-switch 协议。务实中文。"
./B_CONTAINMENT_DESIGN.md:1:# 候选 B [★DEAD 2026-06-24 kill-switch FAIL]: 航拍-地面 ReID 的"物理定向不确定性包含"
./B_CONTAINMENT_DESIGN.md:3:> **⛔ 此方向已死**（零训练 kill-switch 三假设全证伪，cosine A→G 67.41≈训练67.33 sanity过/codex审脚本approve）:
./B_CONTAINMENT_DESIGN.md:8:> 全数据: `cvpb_containment_killswitch_design.md` / `cvpb_containment_full.log`。**下文为原设计, 留作止损记录。** 下步: 3-codex 战略 panel(救援/转向/残酷否决)。
./B_CONTAINMENT_DESIGN.md:10:（2026-06-24, 4-codex 红蓝裁判 panel 完整读后定稿; 全自主决策 —— 后被零训练 kill-switch 推翻）
./B_CONTAINMENT_DESIGN.md:12:## 一、4 codex 收敛结论（全部完整 Read, 0 截断）
./B_CONTAINMENT_DESIGN.md:14:| codex | 角色 | 核心 verdict |
./B_CONTAINMENT_DESIGN.md:17:| v_2 | 红队辩护 | **信心 7/10**(过双 kill-switch→8)。B > C/D(C 退化成"别做局部匹配", D 难证 view-confounder)。headline 必须是"揭示物理定向信息不对称", 不是"我们提出 Gaussian/KL"。**训练版必须 beat avg 52.37**(不只 MaxSim)。 |
./B_CONTAINMENT_DESIGN.md:19:| v_4 | kill-switch 批判 | 裸 kill-switch 不够, 给了 **8 个破坏性对照**。建议**先做加固版 B**。 |
./B_CONTAINMENT_DESIGN.md:33:- vs **PDA**(文本分布⊇图像, 方差=语义范围): 我们方差=**航拍成像欠定度**, 方向由相机高度/分辨率/人体投影面积定, 不是语言粒度。
./B_CONTAINMENT_DESIGN.md:39:## 四、★零训练 kill-switch 协议（加固版, v_4 主导 + v_2/v_3 补）
./B_CONTAINMENT_DESIGN.md:66:## 五、训练版门槛（kill-switch 过后才做, v_2）
./B_CONTAINMENT_DESIGN.md:74:写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
./GOPL_KILLSWITCH_DESIGN.md:1:# GOPL 零训练 kill-switch 设计（2026-06-24, B containment 死后转向; novelty 7/10 存活）
./GOPL_KILLSWITCH_DESIGN.md:13:## 零训练 kill-switch（先验隐藏变量存在 + 证 SMPL 几何独特）
./GOPL_KILLSWITCH_DESIGN.md:42:SMPL 缓存 lab-3090 `/root/work/SOLIDER-REID/cache/smpl_geom/`。occluded_duke `/root/work/SOLIDER-REID/data/occluded_duke`。强 ckpt agent 在 `log/occluded_duke/` 找。复用 error_analysis_geom.py / cvpb_containment_killswitch.py 的 extract/per_query_ap 基建。
./ondisk_pivot.md:14:一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。
./ondisk_pivot.md:16:**硬约束: 只能用 ON-DISK 数据(无下载, 没有视频/tracklet 数据)**: market1501 / MSMT17 / occluded_duke / occluded_reid / occluded_posetrack_reid(image-based) / CARGO(航拍-地面 image) / AG-ReID.v2(航拍-地面 image) / **RSTPReid(text-to-image 行人检索, 有文本标注)** + 强 Swin/SOLIDER backbone(occluded_duke 73 / market 94 / CARGO 67 都有强 ckpt)。
./ondisk_pivot.md:18:问: 用这些 ON-DISK 数据, **单个最可能成的 CCF-B 方法稿方向**是什么? 重点考虑团队没碰过的 **RSTPReid text-to-image** 这条新模态(team 有强 image backbone 可当 visual encoder), 也可考虑跨域(market→occluded_reid)/多数据集泛化/新协议/训练侧创新。要给: 触发观察(在哪能找到一个具体失败) + 重定义 + 机制 + **廉价(最好零训练 frozen-feature)kill-switch** + 撞车核查(text-image ReID 很卷, IRRA/APTM/RDE/CADA 等)。**诚实: 若 text-image 也卷死、其他也都死路, 直说该把 Hubness 写 analysis short 收尾、别再撞。** 联网, 务实中文, verdict + 信心 1-10。
./ondisk_pivot.md:20:我会先核对你们现有实验叙事，再联网查 RSTPReid/text-image ReID 的近年主线和撞车风险。这个问题需要先确认“还能不能争一个方法贡献”，不能靠印象判断。
./ondisk_pivot.md:76:-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
./ondisk_pivot.md:78:-rw-r--r--@  1 abdslm  staff   3928 Jun 24 17:41 hubness_debate.sh
./ondisk_pivot.md:80:-rw-r--r--@  1 abdslm  staff  14500 Jun 24 17:23 novelty_hubness.md
./ondisk_pivot.md:94:web search: RSTPReid text-to-image person re-identification dataset IRRA APTM RDE CADA 2024 2025
./ondisk_pivot.md:96:web search: "APTM" "text-to-image person" RSTPReid
./ondisk_pivot.md:98:web search: "APTM" "RSTPReid" "GitHub"
./ondisk_pivot.md:100:web search: "CADA" "text-image person re-identification"
./ondisk_pivot.md:104:web search: 'RSTPReid' in https://arxiv.org/abs/2308.09911
./ondisk_pivot.md:108:web search: "RDE" "RSTPReID" "57.55"
./ondisk_pivot.md:112:web search: 'RSTPReID dataset' in https://arxiv.org/pdf/2308.09911
./ondisk_pivot.md:116:web search: 'RSTPReid' in https://arxiv.org/pdf/2312.01745
./ondisk_pivot.md:118:web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
./ondisk_pivot.md:120:web search: "RSTPReID" "Rank-1" "2024"
./ondisk_pivot.md:122:web search: "RSTP" "text-based person retrieval" 2024
./ondisk_pivot.md:124:web search: site:arxiv.org "RSTPReid"
./ondisk_pivot.md:126:web search: github IRRA text-to-image person retrieval
./ondisk_pivot.md:134:web search: 'RSTPReid' in https://arxiv.org/pdf/2506.11036
./ondisk_pivot.md:136:联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。
./ondisk_pivot.md:153:一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。
./ondisk_pivot.md:155:**硬约束: 只能用 ON-DISK 数据(无下载, 没有视频/tracklet 数据)**: market1501 / MSMT17 / occluded_duke / occluded_reid / occluded_posetrack_reid(image-based) / CARGO(航拍-地面 image) / AG-ReID.v2(航拍-地面 image) / **RSTPReid(text-to-image 行人检索, 有文本标注)** + 强 Swin/SOLIDER backbone(occluded_duke 73 / market 94 / CARGO 67 都有强 ckpt)。
./ondisk_pivot.md:157:问: 用这些 ON-DISK 数据, **单个最可能成的 CCF-B 方法稿方向**是什么? 重点考虑团队没碰过的 **RSTPReid text-to-image** 这条新模态(team 有强 image backbone 可当 visual encoder), 也可考虑跨域(market→occluded_reid)/多数据集泛化/新协议/训练侧创新。要给: 触发观察(在哪能找到一个具体失败) + 重定义 + 机制 + **廉价(最好零训练 frozen-feature)kill-switch** + 撞车核查(text-image ReID 很卷, IRRA/APTM/RDE/CADA 等)。**诚实: 若 text-image 也卷死、其他也都死路, 直说该把 Hubness 写 analysis short 收尾、别再撞。** 联网, 务实中文, verdict + 信心 1-10。
./ondisk_pivot.md:159:我会先核对你们现有实验叙事，再联网查 RSTPReid/text-image ReID 的近年主线和撞车风险。这个问题需要先确认“还能不能争一个方法贡献”，不能靠印象判断。
./ondisk_pivot.md:215:-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
./ondisk_pivot.md:217:-rw-r--r--@  1 abdslm  staff   3928 Jun 24 17:41 hubness_debate.sh
./ondisk_pivot.md:219:-rw-r--r--@  1 abdslm  staff  14500 Jun 24 17:23 novelty_hubness.md
./ondisk_pivot.md:233:web search: RSTPReid text-to-image person re-identification dataset IRRA APTM RDE CADA 2024 2025
./ondisk_pivot.md:235:web search: "APTM" "text-to-image person" RSTPReid
./ondisk_pivot.md:237:web search: "APTM" "RSTPReid" "GitHub"
./ondisk_pivot.md:239:web search: "CADA" "text-image person re-identification"
./ondisk_pivot.md:243:web search: 'RSTPReid' in https://arxiv.org/abs/2308.09911
./ondisk_pivot.md:247:web search: "RDE" "RSTPReID" "57.55"
./ondisk_pivot.md:251:web search: 'RSTPReID dataset' in https://arxiv.org/pdf/2308.09911
./ondisk_pivot.md:255:web search: 'RSTPReid' in https://arxiv.org/pdf/2312.01745
./ondisk_pivot.md:257:web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
./ondisk_pivot.md:259:web search: "RSTPReID" "Rank-1" "2024"
./ondisk_pivot.md:261:web search: "RSTP" "text-based person retrieval" 2024
./ondisk_pivot.md:263:web search: site:arxiv.org "RSTPReid"
./ondisk_pivot.md:265:web search: github IRRA text-to-image person retrieval
./ondisk_pivot.md:273:web search: 'RSTPReid' in https://arxiv.org/pdf/2506.11036
./ondisk_pivot.md:275:联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。
./ondisk_pivot.md:278:/bin/zsh -lc "sed -n '1,220p' novelty_hubness.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./ondisk_pivot.md:295:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./ondisk_pivot.md:299:web search: person re-identification hubness gallery hub in-degree negative top-k
./ondisk_pivot.md:301:web search: Radovanovic hubness high dimensional retrieval hubs in space 2010
./ondisk_pivot.md:307:web search: "hubness" "Market-1501"
./ondisk_pivot.md:309:web search: "hubness" "vehicle re-identification"
./ondisk_pivot.md:315:web search: Proxy-NCA metric learning proxy anchor loss hard negative mining hubness
./ondisk_pivot.md:323:web search: Mutual Proximity hubness reduction high-dimensional data 2012
./ondisk_pivot.md:325:web search: "Cross-domain similarity local scaling" "hubness" "word translation" "nearest neighbor"
./ondisk_pivot.md:329:web search: "Mutual Proximity" "hubness" "high-dimensional" "nearest neighbor"
./ondisk_pivot.md:355:web search: "popularity bias" "image retrieval" "long-tail" "hubness"
./ondisk_pivot.md:373:结论先给：**Gallery Hubness ReID 有条件存活，不是直接撞车；但“hubness + memory bank + 训练端重加权”在跨模态检索里已经很接近，必须正面引用 HAL / NeighborRetr，不能把 hubness-aware training 当成完全新概念。**  
./ondisk_pivot.md:376:| 类别 | verdict | 撞车判断 | 证据 | 信心 |
./ondisk_pivot.md:378:| (a) 高维检索 hubness / CSLS / MP / local scaling | **存活，但术语不新** | Radovanovic 已把 hubness 定义成 kNN 图里的高 `k-occurrence`，本质就是 in-degree；Schnitzer 的 local/global scaling、CSLS 都是 post-hoc 降 hub。**未见用于 person ReID 训练端 anti-hub**。 | Radovanovic JMLR 定义点在他人 kNN 中出现次数并形成 popular NN；Schnitzer JMLR 讨论对称化 NN 关系减 hub；CSLS 明确为缓解 hubness 的相似度修正。链接：[Radovanovic 2010](https://www.jmlr.org/papers/v11/radovanovic10a.html), [Schnitzer 2012](https://jmlr.org/papers/v13/schnitzer12a.html), [CSLS/Conneau 2017](https://arxiv.org/abs/1710.04087) | 9 |
./ondisk_pivot.md:380:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./ondisk_pivot.md:388:- **外域强撞 HAL / NeighborRetr**：如果我们只写“发现 hubness，用 memory bank 统计 hub，训练时给 hub negative 加权”，会被认为是把 cross-modal retrieval 的 hubness-aware training 移植到 ReID，创新杀伤大。
./ondisk_pivot.md:393:结论先给：**Gallery Hubness ReID 有条件存活，不是直接撞车；但“hubness + memory bank + 训练端重加权”在跨模态检索里已经很接近，必须正面引用 HAL / NeighborRetr，不能把 hubness-aware training 当成完全新概念。**  
./ondisk_pivot.md:396:| 类别 | verdict | 撞车判断 | 证据 | 信心 |
./ondisk_pivot.md:398:| (a) 高维检索 hubness / CSLS / MP / local scaling | **存活，但术语不新** | Radovanovic 已把 hubness 定义成 kNN 图里的高 `k-occurrence`，本质就是 in-degree；Schnitzer 的 local/global scaling、CSLS 都是 post-hoc 降 hub。**未见用于 person ReID 训练端 anti-hub**。 | Radovanovic JMLR 定义点在他人 kNN 中出现次数并形成 popular NN；Schnitzer JMLR 讨论对称化 NN 关系减 hub；CSLS 明确为缓解 hubness 的相似度修正。链接：[Radovanovic 2010](https://www.jmlr.org/papers/v11/radovanovic10a.html), [Schnitzer 2012](https://jmlr.org/papers/v13/schnitzer12a.html), [CSLS/Conneau 2017](https://arxiv.org/abs/1710.04087) | 9 |
./ondisk_pivot.md:400:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./ondisk_pivot.md:408:- **外域强撞 HAL / NeighborRetr**：如果我们只写“发现 hubness，用 memory bank 统计 hub，训练时给 hub negative 加权”，会被认为是把 cross-modal retrieval 的 hubness-aware training 移植到 ReID，创新杀伤大。
./ondisk_pivot.md:428:我们要做 ReID 方法 **OSAC(Open-Set Spectral Over-Collapse)**。重定义: open-set ReID 的隐藏变量是 **seen-ID neural collapse 过强**——训练末期特征过度对齐 seen-ID prototype 几何, unseen-ID 可迁移身份证据低秩化/各向异性化挤到低能谱尾部; k-reciprocal 只能重排恢复不了被压没的维度。hubness 是症状非根因。机制: 训练期 spectral anti-collapse(spectral floor + Top-PC Dropout)保 unseen 证据, 测试仍单 embedding。
./ondisk_pivot.md:430:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'seen-ID 过坍缩定义成 open-set ReID 失败变量 + 训练期抗坍缩保 unseen 迁移': (a)Neural Collapse(Papyan 2020)用到 ReID / open-set / metric learning 没; (b)dimensional collapse / rank collapse in SSL/contrastive(Jing et al, Hua et al); (c)VICReg/Barlow Twins/whitening/decorrelation 泛化零件; (d)ProNet / prototype-projection / classifier-reuse ReID; (e)generalizable/open-set/DG ReID 里有没有'过坍缩伤 unseen 迁移'诊断; (f)feature whitening / group whitening ReID(AMS / SNR / IN 类); (g)spectral/rank regularization metric learning; (h)open-set recognition 里的 feature collapse。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./ondisk_pivot.md:502:web search: "hubness" "person re-identification"
./ondisk_pivot.md:527:**总 verdict：存活，信心 7/10。**  
./ondisk_pivot.md:549:**总 verdict：存活，信心 7/10。**  
./ondisk_pivot.md:566:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./ondisk_pivot.md:567:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./ondisk_pivot.md:577:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./ondisk_pivot.md:589:CLIP-driven fine-grained mining for text-based person search.pdf
./ondisk_pivot.md:595:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./ondisk_pivot.md:599:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./ondisk_pivot.md:642:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./ondisk_pivot.md:652:Internal-External Context Interaction Network for Person Re-Identification.pdf
./ondisk_pivot.md:694:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./ondisk_pivot.md:712:TP-LReID - Lifelong person re-identification using text prompts.pdf
./ondisk_pivot.md:729:本地记录确认了你说的四条 frozen-image 路线已经被更便宜的工具/反证压住；其中 hubness 现在更适合保留为诊断素材。RSTPReid 方向如果要成，不能做“又一个 alignment 模块”，只能做能用零训练先杀的训练侧问题重定义。
./HUBNESS_KILLSWITCH_DESIGN.md:1:# Gallery Hubness ReID 零训练 kill-switch 设计（2026-06-24, B+GOPL 双死后, 战略重评 r_2 主推）
./HUBNESS_KILLSWITCH_DESIGN.md:7:- hubness 在通用高维检索是成熟概念（CSLS/Mutual Proximity 理论可借），但 **ReID 里没人把 gallery 负向 in-degree 定义成失败主变量 + 训练端 anti-hub**。
./HUBNESS_KILLSWITCH_DESIGN.md:8:- vs k-reciprocal re-ranking: 它用 reciprocal neighbor 做 test-time ranking, 不定义 hubness 为失败变量, 不做训练端 anti-hub。
./HUBNESS_KILLSWITCH_DESIGN.md:12:## 零训练 kill-switch（冻结强 Market/MSMT ckpt, 无训练）
./HUBNESS_KILLSWITCH_DESIGN.md:34:Market/MSMT 在 lab-3090 主仓库 `data/`。强 ckpt agent 在 `log/` 找。复用 cvpb_gopl/containment kill-switch 的 extract/per_query_ap 基建。全 frozen + numpy/torch.no_grad。
./all_papers.txt:7:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./all_papers.txt:8:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./all_papers.txt:18:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./all_papers.txt:30:CLIP-driven fine-grained mining for text-based person search.pdf
./all_papers.txt:36:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./all_papers.txt:40:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./all_papers.txt:83:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./all_papers.txt:93:Internal-External Context Interaction Network for Person Re-Identification.pdf
./all_papers.txt:135:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./all_papers.txt:153:TP-LReID - Lifelong person re-identification using text prompts.pdf
./analyses/deep_12.txt:15:**触发观察：** CLIP-ReID 的全局图文对齐能让全局特征变强，但作者用 t-SNE 和 parsing 标注发现，像素级或部位级特征仍然混在一起。身份边界可能清楚，但头、身体、鞋等部位的语义不一致，特别是相似身份的局部特征更难分。
./analyses/deep_12.txt:17:**重定义动作：** 它把问题从“ReID 需要语言监督”推进成“全身级语言监督不足以约束局部语义一致性”。关键词是 within-part semantic inconsistency、part-informed visual-language learning、pixel-level text prompts、fine-grained image-text alignment。它还补了一层冲突定义：human parsing 是身份无关的，而 ReID 要身份相关判别，直接用 parsing 会和身份学习冲突。
./analyses/deep_12.txt:19:**机制怎么长出来：** 既然问题是部位语义不一致，就用 parsing 给每个像素部位名，再构造“某身份的头、鞋、包”等身份感知部位文本提示。既然 parsing 有噪声，就用置信度加权。既然低层和高层特征尺度不同，就加轻量辅助头做多尺度融合对齐。这个机制和重定义绑定很紧，每一块都对应一个被定义出来的障碍。
./analyses/deep_12.txt:23:**reviewer 为什么买账：** 这篇卖的是视角多于模块。真正 novelty 是把 CLIP-ReID 从全局图文对齐改写成“部位级、身份感知、训练期语言监督”，不是简单加 parsing。它的强点是概念链完整：发现局部语义不一致，指出 parsing 与 ReID 的监督冲突，再给出对应机制。
./analyses/deep_12.txt:26:**触发观察：** 文本到图像 ReID 的对比学习默认“非配对样本都是负样本”。作者指出这个假设在 TIReID 里不成立，因为文本描述可能适配多个行人，标注或描述粒度也会造成语义相近的假负样本。把这些样本推远会破坏跨模态对齐。
./analyses/deep_12.txt:28:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./analyses/deep_12.txt:32:**证据闭环：** 它有三类证据：组件消融证明 BGR/TSR/CM-MoC 都有贡献；和 FNE、FNA 这类简单删样本或改标签策略比较，证明“自适应抑制”比硬删除稳；可视化显示高概率假负样本确实和文本语义相近。这个闭环比较完整，因为它直接验证“假负样本存在且会伤训练”。
./cvpb_containment_killswitch_design.md:1:# 候选 B 零训练 kill-switch 设计（cvpb_containment_killswitch.py）
./reassess.sh:8:CTX="一个做行人 ReID 的团队要发 CCF-B **方法稿**, 刚连续用零训练 kill-switch 干净杀掉两个方向:
./reassess.sh:17:ROLES[1]="角色=**残酷判官**。不留情面: 连续2个cheap-kill负 + SMPL反复证伪 + 三大资产全弱, 这个团队到底**还有没有**能发B类方法稿的真路? 还是诚实答案='手里的empirical资产出不了干净B类method, 该换问题域/换数据/换打法'? 联网核查ReID 2024-2026 method空间。**别和稀泥**: 要么明确指1条最可能成的(带理由+廉价kill-switch), 要么明确说'没有, 该转去X'(X具体)。"
./reassess.sh:18:ROLES[2]="角色=**全新方向探子**。彻底放下SMPL/遮挡/航拍-地面三个失败区(不准碰)。团队只剩: 强Swin/SOLIDER backbone(可冻结当强特征源) + occluded_duke/market/MSMT/CARGO数据 + 22招式 + 读过167篇。用联网+招式提**1-2个真正新的B类方向**: 新问题定义或新观察(不是新模块), 每个带触发观察/重定义/机制怎么长/**廉价(最好零训练)kill-switch**/撞车核查。优先'用强backbone冻结特征就能验隐藏变量'的方向(像我们之前那种零训练kill-switch)。务实中文。"
./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
./reviews/deep_23.md:14:你是 ReID 创新方法论拆解员 23。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_23.md:29:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_23.md:36:我会先按项目规则确认当前目标和相关规则是否能读取，然后逐篇用 `pdftotext -f 1 -l 8` 抽取方法部分。后面如果方法部分在前 8 页截断，我会说明依据受限于该命令输出，并用检索补足背景判断。
./reviews/deep_23.md:61:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_23.md:62:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_23.md:72:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_23.md:79:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_23.md:92:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_23.md:96:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_23.md:140:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_23.md:147:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_23.md:190:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_23.md:203:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_23.md:287:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_23.md:300:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_23.md:327:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_23.md:334:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_23.md:491:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_23.md:512:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_23.md:528:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_23.md:586:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_23.md:587:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_23.md:613:/bin/zsh -lc 'pdftotext -v' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:615:pdftotext version 26.04.0
./reviews/deep_23.md:630:/bin/zsh -lc "pdftotext -f 1 -l 8 'RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:664:misalignment between the visual and textual features caused by occlusions. These challenges motivate us
./reviews/deep_23.md:670:the textual embedding space to enhance cross-modal alignment, and a multi-branch distribution alignment
./reviews/deep_23.md:695:as CLIP [4], enable joint modeling of visual features and their corresponding high-level textual descriptions through large-scale contrastive
./reviews/deep_23.md:698:by incorporating learnable textual tokens to complement visual representations. While this approach demonstrates strong performance
./reviews/deep_23.md:703:them with complementary textual information. (2) Limited geometric
./reviews/deep_23.md:714:0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_23.md:734:enhancement in the text feature space. By applying contrastive loss to
./reviews/deep_23.md:788:masking enhancement in the text feature space, alleviating the
./reviews/deep_23.md:816:leveraging large-scale image–text pairs. These models show strong
./reviews/deep_23.md:819:align visual and textual features through prompt-based supervision
./reviews/deep_23.md:820:in the global embedding space, which improves identity-level semantic consistency but remains limited in associating textual cues with
./reviews/deep_23.md:826:an inter-text contrastive constraint to enhance the robustness of IDconditioned text prototypes under missing semantic components.
./reviews/deep_23.md:835:in the first stage, the parameters of the text encoder and image encoder
./reviews/deep_23.md:850:contrastive reconstruction within the text space. Specifically, in the first
./reviews/deep_23.md:851:training stage, we apply double masking to the learnable text prompt
./reviews/deep_23.md:857:where 𝑦̂ represents the masked text prompt, Mask is the masking
./reviews/deep_23.md:860:from the same text form positive pairs, while those from different texts
./reviews/deep_23.md:862:original prompt is then encoded by the text encoder, and its feature
./reviews/deep_23.md:867:𝑠(𝑉𝑖 , 𝑇𝑖 ) = 𝑉𝑖 ⋅ 𝑇𝑖 = 𝑔𝑉 (img𝑖 ) ⋅ 𝑔𝑇 (text 𝑖 ),
./reviews/deep_23.md:891:where 𝑔𝑉 (⋅) and 𝑔𝑇 (⋅) represent the image encoder and text encoder
./reviews/deep_23.md:892:respectively. The image-to-text and text-to-image contrastive loss are
./reviews/deep_23.md:904:where 𝑉 is the image feature embedding, 𝑇 is the text feature embedding, 𝑠(⋅, ⋅) represents the inner product similarity calculation, and 𝐵
./reviews/deep_23.md:905:denotes the batch size. Next, the original text prompts are enhanced
./reviews/deep_23.md:908:text encoder to obtain the corresponding masked feature, which is
./reviews/deep_23.md:909:further refined using an inter-text contrastive loss. The contrastive loss
./reviews/deep_23.md:931:where 𝑇𝑖 and 𝑇𝑖′ denote two text features of the same identity obtained by applying the dual-masked prompt augmentation twice, and
./reviews/deep_23.md:932:𝑗 enumerates all candidate text features within the mini-batch. 𝑁
./reviews/deep_23.md:933:denotes the number of text prototypes involved in the batch for text-totext contrastive learning. 𝑠(⋅, ⋅) is the cosine similarity function in the
./reviews/deep_23.md:966:space. In CLIP-style ReID, the learnable text prompt acts as an identityconditioned semantic prototype for cross-modal alignment. By applying dual masking to the prompt tokens, DMPA introduces structured
./reviews/deep_23.md:967:missing-information perturbations on the text side and enforces the
./reviews/deep_23.md:968:resulting text features to remain semantically consistent through the
./reviews/deep_23.md:969:inter-text contrastive loss. This mechanism reduces the co-adaptation
./reviews/deep_23.md:970:of prompt dimensions and encourages the text prototype to remain
./reviews/deep_23.md:971:discriminative even when part of the semantic context is removed.
./reviews/deep_23.md:973:evidence, can still be aligned with a stable text-side semantic anchor,
./reviews/deep_23.md:1024:image–text alignment and lack explicit local feature enhancement
./reviews/deep_23.md:1129:from image to text is defined as:
./reviews/deep_23.md:1585:The image and text encoders in the proposed RMPSNet both adopt
./reviews/deep_23.md:1590:64. In Stage 1, only learnable text prompts participate in the training.
./reviews/deep_23.md:1591:The parameters of both the text encoder and the image encoder are
./reviews/deep_23.md:1668:several key factors. First, the DMPA module enables fine-grained crossmodal alignment, allowing textual semantics to effectively complement
./reviews/deep_23.md:1681:optimization tension when jointly trained. Specifically, DMPA encourages semantic invariance on the text side by stabilizing ID-conditioned
./reviews/deep_23.md:1807:conditions. By leveraging partial textual descriptions and enforcing
./reviews/deep_23.md:2012:/bin/zsh -lc "pdftotext -f 1 -l 8 'Robust mixed-degradation person Re-identification via structural consistency distillation.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:3311:前两篇已经读到核心：一篇把遮挡解释成真实遮挡分布、文本语义缺失和增强后特征分布不一致三件事；另一篇把退化 ReID 从“单一退化域偏移”改写成“混合退化下的分布偏移加身份拓扑扭曲”。我会补一下被输出截断的中间方法细节，然后继续后四篇。
./reviews/deep_23.md:3325:/bin/zsh -lc "pdftotext -f 3 -l 5 'RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:3331:leveraging large-scale image–text pairs. These models show strong
./reviews/deep_23.md:3334:align visual and textual features through prompt-based supervision
./reviews/deep_23.md:3335:in the global embedding space, which improves identity-level semantic consistency but remains limited in associating textual cues with
./reviews/deep_23.md:3341:an inter-text contrastive constraint to enhance the robustness of IDconditioned text prototypes under missing semantic components.
./reviews/deep_23.md:3350:in the first stage, the parameters of the text encoder and image encoder
./reviews/deep_23.md:3365:contrastive reconstruction within the text space. Specifically, in the first
./reviews/deep_23.md:3366:training stage, we apply double masking to the learnable text prompt
./reviews/deep_23.md:3372:where 𝑦̂ represents the masked text prompt, Mask is the masking
./reviews/deep_23.md:3375:from the same text form positive pairs, while those from different texts
./reviews/deep_23.md:3377:original prompt is then encoded by the text encoder, and its feature
./reviews/deep_23.md:3382:𝑠(𝑉𝑖 , 𝑇𝑖 ) = 𝑉𝑖 ⋅ 𝑇𝑖 = 𝑔𝑉 (img𝑖 ) ⋅ 𝑔𝑇 (text 𝑖 ),
./reviews/deep_23.md:3406:where 𝑔𝑉 (⋅) and 𝑔𝑇 (⋅) represent the image encoder and text encoder
./reviews/deep_23.md:3407:respectively. The image-to-text and text-to-image contrastive loss are
./reviews/deep_23.md:3419:where 𝑉 is the image feature embedding, 𝑇 is the text feature embedding, 𝑠(⋅, ⋅) represents the inner product similarity calculation, and 𝐵
./reviews/deep_23.md:3420:denotes the batch size. Next, the original text prompts are enhanced
./reviews/deep_23.md:3423:text encoder to obtain the corresponding masked feature, which is
./reviews/deep_23.md:3424:further refined using an inter-text contrastive loss. The contrastive loss
./reviews/deep_23.md:3446:where 𝑇𝑖 and 𝑇𝑖′ denote two text features of the same identity obtained by applying the dual-masked prompt augmentation twice, and
./reviews/deep_23.md:3447:𝑗 enumerates all candidate text features within the mini-batch. 𝑁
./reviews/deep_23.md:3448:denotes the number of text prototypes involved in the batch for text-totext contrastive learning. 𝑠(⋅, ⋅) is the cosine similarity function in the
./reviews/deep_23.md:3481:space. In CLIP-style ReID, the learnable text prompt acts as an identityconditioned semantic prototype for cross-modal alignment. By applying dual masking to the prompt tokens, DMPA introduces structured
./reviews/deep_23.md:3482:missing-information perturbations on the text side and enforces the
./reviews/deep_23.md:3483:resulting text features to remain semantically consistent through the
./reviews/deep_23.md:3484:inter-text contrastive loss. This mechanism reduces the co-adaptation
./reviews/deep_23.md:3485:of prompt dimensions and encourages the text prototype to remain
./reviews/deep_23.md:3486:discriminative even when part of the semantic context is removed.
./reviews/deep_23.md:3488:evidence, can still be aligned with a stable text-side semantic anchor,
./reviews/deep_23.md:3539:image–text alignment and lack explicit local feature enhancement
./reviews/deep_23.md:3644:from image to text is defined as:
./reviews/deep_23.md:3681:/bin/zsh -lc "pdftotext -f 1 -l 8 'See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:3687:See What You Seek: Semantic Contextual Integration
./reviews/deep_23.md:3701:framework Semantic Contextual Integration (SCI), which leverages
./reviews/deep_23.md:3702:the visual-textual representation capabilities of CLIP to reduce
./reviews/deep_23.md:3705:which employs dual learnable text tokens to disentangle clothingrelated semantics from confounding factors, thereby isolating
./reviews/deep_23.md:3707:Interaction Module (SIM) that uses orthogonalized text features to
./reviews/deep_23.md:3711:context with high-dimensional insights. Extensive experiments on
./reviews/deep_23.md:3776:the appearance of clothing, e.g., texture and color, to distinguish
./reviews/deep_23.md:3828:language, yielding context-aware representations beneficial for
./reviews/deep_23.md:3835:In this paper, we introduce the Semantic Contextual Integration (SCI) framework for CC-ReID. SCI uses CLIP
./reviews/deep_23.md:3838:negative factors at the text level while preserving key positive
./reviews/deep_23.md:3840:text tokens to disentangle confounding semantics (both positive
./reviews/deep_23.md:3843:text features guide visual encoding via our Semantic-Guided
./reviews/deep_23.md:3846:with high-dimensional semantic context and achieves superior
./reviews/deep_23.md:3849:• We propose the Semantic Contextual Integration (SCI)
./reviews/deep_23.md:3854:module to filter and refine text-level features, improving
./reviews/deep_23.md:3857:to guide visual representations using refined text features,
./reviews/deep_23.md:3893:CCFA [43] augment clothing color and texture diversity;
./reviews/deep_23.md:3900:appearance, such as color and texture, to extract discriminative relations; and DCLR [47] synthesizes multi-clothing images
./reviews/deep_23.md:3901:features. Feature representation learning in this context can be via diffusion and merges them into training data.
./reviews/deep_23.md:3905:single feature vector per image [17], while local methods aggre- we integrate visual and textual modalities via CLIP to enrich
./reviews/deep_23.md:3908:attributes [20] or synthesized samples [21], to enrich context. C. Vision-Language Learning
./reviews/deep_23.md:3911:TransReID [23]. ISP [24] further refines alignment by locating images with text, improving downstream visual tasks. CLIP [11]
./reviews/deep_23.md:3912:body parts and carried items at the pixel level. Moreover, uses paired image and text encoders to learn a shared embedsince person ReID is inherently formulated as a ranking ding space via contrastive learning, benefiting applications such
./reviews/deep_23.md:3915:Prompt learning extends CLIP by making context tokens
./reviews/deep_23.md:4049:Interaction Module (SIM). SSE mitigates clothing bias by removing negative semantic factors, while SIM employs the refined text features to guide visual
./reviews/deep_23.md:4052:image, and DenseCLIP [50] applies pixel-text matching for
./reviews/deep_23.md:4059:body parts for occluded ReID. VGSG [54] groups text features
./reviews/deep_23.md:4061:integrates CLIP-generated textual information into a novel
./reviews/deep_23.md:4067:large-scale image-text training captures high-level semantics
./reviews/deep_23.md:4068:(e.g., body shape and context) remain invariant to clothing. We
./reviews/deep_23.md:4074:joint image-text representations via a contrastive objective. It
./reviews/deep_23.md:4075:comprises a visual encoder I(·) and a text encoder T (·), whose
./reviews/deep_23.md:4079:: Dual textual learnable vectors promptid ,
./reviews/deep_23.md:4082:Output : Optimized text representations Fort
./reviews/deep_23.md:4088:encoder I(·) Train context vectors for dual prompts
./reviews/deep_23.md:4092:Compute text features Fid
./reviews/deep_23.md:4112:zero-shot prompts by introducing learnable context tokens.
./reviews/deep_23.md:4115:vector and [CLS] is a class token, CLIP computes image-text
./reviews/deep_23.md:4134:high-dimensional text features, preserving only ID-relevant
./reviews/deep_23.md:4138:: Dual textual learnable vectors promptid ,
./reviews/deep_23.md:4141:Output : Optimized text representations Fort
./reviews/deep_23.md:4146:Extract dual text features Ftxt using the frozen text
./reviews/deep_23.md:4147:encoder T (·) Compute orthogonalized text features
./reviews/deep_23.md:4152:contextual information using Eq. (13) Compute
./reviews/deep_23.md:4166:text encoder T (·) to extract image and dual text features,
./reviews/deep_23.md:4168:the text tokens [X]i . This allows us to learn contextual
./reviews/deep_23.md:4169:representations by updating the text tokens, thereby acquiring
./reviews/deep_23.md:4170:distinct textual representations for each ID and their clothing:
./reviews/deep_23.md:4184:Finally, we calculate the image-to-text contrastive loss Li2t :
./reviews/deep_23.md:4200:represent the ID-specific text features where Vi and Fort (i) are paired visual and text embeddings,
./reviews/deep_23.md:4202:and clothing text features, respectively.
./reviews/deep_23.md:4204:After extracting the dual text representations, we project Fclo Since multiple images in a batch may belong to the same ID,
./reviews/deep_23.md:4207:to reduce the influence of clothing and emphasize meaning there may be multiple positive samples, the text-tothe implicit positive factors. This step clarifies the relation be- image contrastive loss Lt2i (yi ) is calculated as:
./reviews/deep_23.md:4263:To optimize the text features, we design a loss function
./reviews/deep_23.md:4298:the batch, and Tyi is the text embedding corresponding to label
./reviews/deep_23.md:4308:in visual or textual features improves performance in clothchanging scenarios. However, these methods typically treat
./reviews/deep_23.md:4309:visual and textual branches independently, lacking interaction. Our method refines visual representations by leveraging
./reviews/deep_23.md:4310:clothing-irrelevant textual features, enhancing the interaction
./reviews/deep_23.md:4311:between visual and textual branches. This integration introduces
./reviews/deep_23.md:4315:As illustrated in Fig. 3, we employ a Transformer decoder [58] to model interactions between visual and textual data.
./reviews/deep_23.md:4381:feature extraction to align features with the relevant semantic context.
./reviews/deep_23.md:4408:captures global context information.
./reviews/deep_23.md:4422:feature map obtained by adding this context-aware feature to
./reviews/deep_23.md:4425:leverage textual embeddings to generate attention weights over
./reviews/deep_23.md:4428:more discriminative and context-aware visual representations:
./reviews/deep_23.md:4455:visual features under the guidance of textual embeddings, and
./reviews/deep_23.md:4457:aligned with the semantic content of the text.
./reviews/deep_23.md:4459:that the visual features are enriched with relevant semantic information from the text, leading to more robust and contextually
./reviews/deep_23.md:4472:where text embeddings guide the visual encoding process. By
./reviews/deep_23.md:4501:image-to-text cross-entropy loss Li2tce as:
./reviews/deep_23.md:4527:while aligning visual and textual representations. The weights
./reviews/deep_23.md:4606:the text feature dimension (scaled from 512 to 1024). For the [X]1 . . . [X]M clothes”) to learn features separately at
./reviews/deep_23.md:4607:transformer branch, we employ 12 transformer layers with the textual level. We observe consistent improvements of
./reviews/deep_23.md:4609:dimension from 768 to 512 to align with the text features.
./reviews/deep_23.md:4614:These results indicate: 1) The original generic textual
./reviews/deep_23.md:4617:epochs (ViT-B/16), starting at 3.5 × 10−4 and reducing by a cloth-relevant aspects. 2) Separating these features at the textual
./reviews/deep_23.md:4634:IMS+GEP [46], AFL [4], CVSL [36], MCSC [45], MAL-F [41], the SSE module generates final textual features that guide
./reviews/deep_23.md:4640:textual representations, many ID-irrelevant negative factors,
./reviews/deep_23.md:5135:prompts are trained while the visual and text encoders remain
./reviews/deep_23.md:5140:the text-image similarity computations.
./reviews/deep_23.md:5260:illustrate how the feature extraction model highlights various dual textual prompts and orthogonalization in enhancing feature
./reviews/deep_23.md:5266:knees. If people wear long pants, this focus could potentially leveraging textual information obtained from the SSE module
./reviews/deep_23.md:5271:the similarity distributions of features between the text prompts in Fig. 8(a) and (b), thereby outperforming the self-attention
./reviews/deep_23.md:5274:This underscores the non-local mechanism’s enhanced casimilarity remains dispersed. Fig. 7(b) shows the distribution pability to capture comprehensive global context, which is
./reviews/deep_23.md:5280:/bin/zsh -lc "pdftotext -f 3 -l 5 'Robust mixed-degradation person Re-identification via structural consistency distillation.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:5372:injecting purified textual cues into visual feature learning.
./reviews/deep_23.md:5659:/bin/zsh -lc "pdftotext -f 4 -l 5 'See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:5673:high-dimensional text features, preserving only ID-relevant
./reviews/deep_23.md:5677:: Dual textual learnable vectors promptid ,
./reviews/deep_23.md:5680:Output : Optimized text representations Fort
./reviews/deep_23.md:5685:Extract dual text features Ftxt using the frozen text
./reviews/deep_23.md:5686:encoder T (·) Compute orthogonalized text features
./reviews/deep_23.md:5691:contextual information using Eq. (13) Compute
./reviews/deep_23.md:5705:text encoder T (·) to extract image and dual text features,
./reviews/deep_23.md:5707:the text tokens [X]i . This allows us to learn contextual
./reviews/deep_23.md:5708:representations by updating the text tokens, thereby acquiring
./reviews/deep_23.md:5709:distinct textual representations for each ID and their clothing:
./reviews/deep_23.md:5723:Finally, we calculate the image-to-text contrastive loss Li2t :
./reviews/deep_23.md:5739:represent the ID-specific text features where Vi and Fort (i) are paired visual and text embeddings,
./reviews/deep_23.md:5741:and clothing text features, respectively.
./reviews/deep_23.md:5743:After extracting the dual text representations, we project Fclo Since multiple images in a batch may belong to the same ID,
./reviews/deep_23.md:5746:to reduce the influence of clothing and emphasize meaning there may be multiple positive samples, the text-tothe implicit positive factors. This step clarifies the relation be- image contrastive loss Lt2i (yi ) is calculated as:
./reviews/deep_23.md:5802:To optimize the text features, we design a loss function
./reviews/deep_23.md:5837:the batch, and Tyi is the text embedding corresponding to label
./reviews/deep_23.md:5847:in visual or textual features improves performance in clothchanging scenarios. However, these methods typically treat
./reviews/deep_23.md:5848:visual and textual branches independently, lacking interaction. Our method refines visual representations by leveraging
./reviews/deep_23.md:5849:clothing-irrelevant textual features, enhancing the interaction
./reviews/deep_23.md:5850:between visual and textual branches. This integration introduces
./reviews/deep_23.md:5854:As illustrated in Fig. 3, we employ a Transformer decoder [58] to model interactions between visual and textual data.
./reviews/deep_23.md:5920:feature extraction to align features with the relevant semantic context.
./reviews/deep_23.md:5947:captures global context information.
./reviews/deep_23.md:5961:feature map obtained by adding this context-aware feature to
./reviews/deep_23.md:5964:leverage textual embeddings to generate attention weights over
./reviews/deep_23.md:5967:more discriminative and context-aware visual representations:
./reviews/deep_23.md:5994:visual features under the guidance of textual embeddings, and
./reviews/deep_23.md:5996:aligned with the semantic content of the text.
./reviews/deep_23.md:5998:that the visual features are enriched with relevant semantic information from the text, leading to more robust and contextually
./reviews/deep_23.md:6011:where text embeddings guide the visual encoding process. By
./reviews/deep_23.md:6040:image-to-text cross-entropy loss Li2tce as:
./reviews/deep_23.md:6066:while aligning visual and textual representations. The weights
./reviews/deep_23.md:6099:/bin/zsh -lc "pdftotext -f 1 -l 8 'Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:6149:patches independently and thereby overlooking broader contextual cues necessary for precise alignment. This exacerbates
./reviews/deep_23.md:6248:though powerful in global context modeling, may dilute such
./reviews/deep_23.md:6252:contextual information that benefits the subsequent ReID tasks.
./reviews/deep_23.md:6348:alignment when the local contextual cues are more consolidated. Ablation studies in Section IV-D show that this dynamic
./reviews/deep_23.md:6787:context, both Transformer variants emphasize global attention,
./reviews/deep_23.md:6928:/bin/zsh -lc "pdftotext -f 1 -l 8 'Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:7036:1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
./reviews/deep_23.md:7111:• Lack of historical relevance. In the context of learning
./reviews/deep_23.md:7204:and texture are indispensable in improving retrieval performance [41], [42], [43]. Therefore, fully understanding and
./reviews/deep_23.md:7783:pattern alignment network (MPANet) to mine the subtle texture differences for VI-ReID. MPANet introduces
./reviews/deep_23.md:7813:/bin/zsh -lc "pdftotext -f 1 -l 8 'Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:7866:especially salient colour and texture information of clothes.
./reviews/deep_23.md:8476:Xing Sun, and Wei-Shi Zheng. Learning 3d shape feature for textureinsensitive person re-identification. In Proceedings of the IEEE/CVF
./reviews/deep_23.md:8623:/bin/zsh -lc "pdftotext -f 8 -l 12 'Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_23.md:8672:pattern alignment network (MPANet) to mine the subtle texture differences for VI-ReID. MPANet introduces
./reviews/deep_23.md:8953:1. **触发观察**：作者抓住三个具体失败点。遮挡会造成视觉语义缺失，CLIP-ReID 的图文全局对齐在遮挡下会错位，常用随机擦除默认遮挡均匀分布，但真实遮挡更多集中在下半身，尤其腿部。
./reviews/deep_23.md:8955:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reviews/deep_23.md:8956:4. **证据闭环**：关键不是只报总成绩，而是三类消融分别对应三类重定义。`Baseline+RPE` 在 Occluded-Duke 上大幅超过普通随机擦除，证明“遮挡分布先验”有用；`Baseline+DMPA` 和去掉 DMPA 的版本证明文本语义稳态不是装饰；MDO 的 only enhancement 和完整 MDO 对比证明两步对抗比普通扰动更有效。
./reviews/deep_23.md:8966:**See What You Seek: Semantic Contextual Integration**
./reviews/deep_23.md:8968:2. **重定义动作**：它把换衣问题从“消除衣服”改写成“分离负语义并显式挖出正语义”。关键词是 `negative factors`、`positive factors`、`semantic contextual integration`、`semantic separation`、`orthogonalized text features`。
./reviews/deep_23.md:8969:3. **机制怎么长出来**：SSE 用两个 prompt，一个描述 person，一个描述 clothes。先得到身份文本特征和衣服文本特征，再把衣服方向从身份方向里投影并减掉，得到去衣服后的语义特征。SIM 再让这个正交化文本特征通过非局部操作和跨注意力去指导视觉特征。这个设计和“负语义分离、正语义引导”的重定义绑定很紧。
./reviews/deep_23.md:8970:4. **证据闭环**：它的关键消融是 SSE 单独有效，SIM 单独在 CNN 上甚至会伤性能，SSE 加 SIM 才最好。这证明“文本指导本身不够，必须先把衣服语义分离干净”。t-SNE 更紧凑、prompt 引导热图更关注头肩、下身、鞋，特征相似度分布在正交化后更清晰，这些可视化都在支撑“语义分离是对的”。
./reviews/deep_23.md:8971:5. **审稿人为什么买账**：这篇卖的是视角。CLIP-ReID 已经有 ID prompt，CAL 已经有衣服对抗，但它把换衣讲成“负因素和正因素同时存在”的语义问题，用文本空间正交化给出可解释动作。新意不是“用了 CLIP”，而是把衣服偏置从语义空间里切出来，再反过来指导视觉。
./reviews/deep_23.md:9003:1. **触发观察**：作者抓住三个具体失败点。遮挡会造成视觉语义缺失，CLIP-ReID 的图文全局对齐在遮挡下会错位，常用随机擦除默认遮挡均匀分布，但真实遮挡更多集中在下半身，尤其腿部。
./reviews/deep_23.md:9005:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reviews/deep_23.md:9006:4. **证据闭环**：关键不是只报总成绩，而是三类消融分别对应三类重定义。`Baseline+RPE` 在 Occluded-Duke 上大幅超过普通随机擦除，证明“遮挡分布先验”有用；`Baseline+DMPA` 和去掉 DMPA 的版本证明文本语义稳态不是装饰；MDO 的 only enhancement 和完整 MDO 对比证明两步对抗比普通扰动更有效。
./reviews/deep_23.md:9016:**See What You Seek: Semantic Contextual Integration**
./reviews/deep_23.md:9018:2. **重定义动作**：它把换衣问题从“消除衣服”改写成“分离负语义并显式挖出正语义”。关键词是 `negative factors`、`positive factors`、`semantic contextual integration`、`semantic separation`、`orthogonalized text features`。
./reviews/deep_23.md:9019:3. **机制怎么长出来**：SSE 用两个 prompt，一个描述 person，一个描述 clothes。先得到身份文本特征和衣服文本特征，再把衣服方向从身份方向里投影并减掉，得到去衣服后的语义特征。SIM 再让这个正交化文本特征通过非局部操作和跨注意力去指导视觉特征。这个设计和“负语义分离、正语义引导”的重定义绑定很紧。
./reviews/deep_23.md:9020:4. **证据闭环**：它的关键消融是 SSE 单独有效，SIM 单独在 CNN 上甚至会伤性能，SSE 加 SIM 才最好。这证明“文本指导本身不够，必须先把衣服语义分离干净”。t-SNE 更紧凑、prompt 引导热图更关注头肩、下身、鞋，特征相似度分布在正交化后更清晰，这些可视化都在支撑“语义分离是对的”。
./reviews/deep_23.md:9021:5. **审稿人为什么买账**：这篇卖的是视角。CLIP-ReID 已经有 ID prompt，CAL 已经有衣服对抗，但它把换衣讲成“负因素和正因素同时存在”的语义问题，用文本空间正交化给出可解释动作。新意不是“用了 CLIP”，而是把衣服偏置从语义空间里切出来，再反过来指导视觉。
./reviews/deep_28.md:14:你是 ReID 创新方法论拆解员 28。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_28.md:35:我会先按项目规则确认当前目标和工作区纪律，然后直接用 `pdftotext` 抽取这 5 篇的前 8 页方法部分。这个任务涉及 PDF 阅读，我会只取和拆解方法论相关的文本，不改任何文件。
./reviews/deep_28.md:71:/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:80:# PDF Skill
./reviews/deep_28.md:94:3. Use `pdfplumber` or `pypdf` for text extraction and quick checks; do not rely on text extraction for layout fidelity.
./reviews/deep_28.md:149:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_28.md:261:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_28.md:282:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_28.md:298:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_28.md:356:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_28.md:357:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_28.md:396:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_28.md:397:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_28.md:407:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_28.md:414:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_28.md:427:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_28.md:431:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_28.md:475:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_28.md:482:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_28.md:525:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_28.md:538:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_28.md:642:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_28.md:655:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_28.md:682:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_28.md:689:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_28.md:798:Keywords:        visible-infrared person re-identification, text prompts, knowledge distillation
./reviews/deep_28.md:800:Creator:         LaTeX with acmart 2024/12/28 v2.12 Typesetting articles for the Association for Computing Machinery and hyperref 2024-01-20 v7.01h Hypertext links for LaTeX
./reviews/deep_28.md:825:PDF 技能的建议是需要版式时再渲染；这次用户明确要方法论文本，所以我按指定的 `pdftotext -f 1 -l 8` 走。接下来先定位每篇的方法标题和实验标题，避免只读摘要或漏掉方法段。
./reviews/deep_28.md:867:第一，做训练前两轮独立审查里的 Codex 这一轮。一个改了方法的实验，在开始训练之前，主 agent 会用 `codex exec` 起你来做一次独立代码审查。你看不到另一个审查者的结论，也不知道这是第几轮，也不要假设自己知道改了什么。审查范围是设计文档、新增和改动代码、配置、对照组变量隔离、评测协议和数据流。
./reviews/deep_28.md:869:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reviews/deep_28.md:885:凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_28.md:903:自己写的评测脚本或分析脚本，要先用它复现一个已知的基线成绩。对得上之后，才能用它的结果下结论。新的评测口径、新的度量、新的脚本都按这条执行。
./reviews/deep_28.md:972:任何改了模型或者有新设计的实验，在启动训练之前，都要经过两轮互不通气的独立审查。一轮由 Claude 做，一轮由 Codex 做。两个审查者互相看不到对方结论，也不知道这是第几轮。
./reviews/deep_28.md:974:Codex 这一轮要完整审一遍，不是只看某几处。要看设计文档、新增和改动的代码、配置、对照组、变量隔离和评测协议。结论写进 `experiments/exp{编号}/review-codex.md`。
./reviews/deep_28.md:1017:重大决策前，尽量让两个独立视角辩一辩。一个为方案 A 辩护，一个为方案 B 辩护，从技术可行性、创新性、论文价值、风险、成本几方面讲，并给出信心分。最后综合判断，把结论写进决策记录。
./reviews/deep_28.md:1047:绝对不要用宽泛匹配的 `pkill -f codex`，那会误杀 Mac 上的 Codex.app。要杀进程就精确匹配。
./reviews/deep_28.md:1061:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reviews/deep_28.md:1124:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:1152:/bin/zsh -lc "pdftotext -f 1 -l 8 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:1184:1059:34, 054 images and 68, 126 texts. Both the validation set and
./reviews/deep_28.md:1185:1061:078 images and 6, 158 texts, and the latter has 3, 074 images
./reviews/deep_28.md:1192:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:1200:114:2𝑖 + 1). Using 𝑊𝑡𝑒𝑥𝑡 (𝑊𝑖𝑚𝑔 ) to represent the parameters of text
./reviews/deep_28.md:1403:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | nl -ba | sed -n '79,613p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:1416:    90	strategy which alternately optimizes the text prompts and image
./reviews/deep_28.md:1418:    92	encoders in different epochs. When optimizing text prompts, we
./reviews/deep_28.md:1440:   114	2𝑖 + 1). Using 𝑊𝑡𝑒𝑥𝑡 (𝑊𝑖𝑚𝑔 ) to represent the parameters of text
./reviews/deep_28.md:1441:   115	prompts (image encoders) and 𝐹𝑡𝑒𝑥𝑡 (𝐹𝑖𝑚𝑔 ) to represent the optimization process of text prompts (image encoders), the optimization process can be formalized as:
./reviews/deep_28.md:1512:   186	optimizing text prompts and the modality alignment loss is used
./reviews/deep_28.md:1522:   196	modality alignment loss. The loss treats the text features as a special intermediate modality, and aligns both visible and infrared
./reviews/deep_28.md:1523:   197	modalities to the text modality. In this way, prompts can serve
./reviews/deep_28.md:1526:   200	help of MPT, text prompts can help image encoders become better,
./reviews/deep_28.md:1529:   203	both text prompts and image encoders will continue to improve.
./reviews/deep_28.md:1560:   234	each other. Under such a premise, we propose the modality knowledge propagation loss. The loss constrains the predicted class probability of text features to be similar to that of image features. In
./reviews/deep_28.md:1561:   235	this way, person-related knowledge is propagated from image encoders into text prompts. The loss function can be expressed as:
./reviews/deep_28.md:1566:   240	to generate text prompts, but such strategy can only generate prompts
./reviews/deep_28.md:1573:   247	structure of MPT is shown in the Fig. 2. Specifically, the text encoder takes prompts as input and outputs corresponding text features, while image encoders take images as input and output corresponding image features. When 𝑒𝑝𝑜𝑐ℎ = 2𝑖, we fix image encoders and optimize prompts with modality knowledge propagation loss. And when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 + 1, we fix prompts and optimize
./reviews/deep_28.md:1599:   273	is the text feature with the 𝑖-th identity (each identity only has
./reviews/deep_28.md:1600:   274	one text prompt, so there is one text feature for each identity). 𝐹 𝐶
./reviews/deep_28.md:1609:   283	Due to the fact that the high-level semantics contained in the text
./reviews/deep_28.md:1611:   285	we can consider optimized prompts as a special intermediate modality. When optimizing encoders, we treat the text features generated by prompts as the “bridge” and use them to connect the two
./reviews/deep_28.md:1613:   287	The proposed modality alignment loss first calculates the similarity between text features and image features, then utilizes contrastive learning loss to align the image features of both modalities
./reviews/deep_28.md:1842:   516	to the text features. Specifically, before calculating similarity, we
./reviews/deep_28.md:1844:   518	similarity between text and image features can be expressed as:
./reviews/deep_28.md:1871:   545	number of text prompts. 𝑞𝑘 is the value in the target distribution,
./reviews/deep_28.md:1892:   566	With the help of the above loss functions, text prompts and image encoders can mutually promote each other, ultimately improving the performance of cross-modality retrieval.
./reviews/deep_28.md:1942:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '51,430p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:2326:/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*|.*Dataset.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:2469:/bin/zsh -lc "pdftotext -f 1 -l 8 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '399,1058p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:2546:   474	and text encoder in Section IV-A. Then, the architecture of
./reviews/deep_28.md:2552:   480	with corresponding text queries T = {T1 , T2 , . . . , Tn }, textbased person image retrieval is carried out by evaluating the
./reviews/deep_28.md:2553:   481	similarity between each text query and every image, and then
./reviews/deep_28.md:2718:   646	Fig. 2. Overview of the proposed UP-Person framework. Left is the overall backbone of UP-Person, which consists of image encoder and text encoder
./reviews/deep_28.md:2721:   649	transformer block for both image and text encoders. In addition to prefix tokens in the keys and values of MHA, S-Prefix proposes a Sp factor in attention
./reviews/deep_28.md:2753:   681	as text encoder, which is also a 12-layer transformer. The
./reviews/deep_28.md:2755:   683	of the image encoder. For the input text T , we tokenize the
./reviews/deep_28.md:2775:   703	parts of the person image or text description by learnable prefix
./reviews/deep_28.md:2782:   710	in text-based person retrieval.
./reviews/deep_28.md:2791:   719	LoRA and Adapter within image and text encoders. (1)
./reviews/deep_28.md:2793:   721	the adaptation ability of the prefix embeddings for text-based
./reviews/deep_28.md:3080:  1008	embeddings for image-text pairs into the KL divergence to
./reviews/deep_28.md:3082:  1010	For a mini-batch containing N image-text
./reviews/deep_28.md:3086:  1014	set of image-text representation pairs as fiv , fjt , yi,j j=1 ,
./reviews/deep_28.md:3099:  1027	where sim(f v , f t ) denotes cosine similarity between text embedding f v and image embedding f t , and τ is a temperature
./reviews/deep_28.md:3103:  1031	matching probability. Finally, the SDM loss from image to text
./reviews/deep_28.md:3122:  1050	where Li2t denotes image-to-text matching for the input text,
./reviews/deep_28.md:3123:  1051	and Lt2i denotes text-to-image matching for the input image,
./reviews/deep_28.md:3129:  1057	CUHK-PEDES as the most commonly used dataset, contains 40,206 images and 80,412 textual descriptions for 13,003
./reviews/deep_28.md:3133:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*|.*Dataset.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:3142:90:There are three primary causes of noisy labels in this context:
./reviews/deep_28.md:3143:115:1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
./reviews/deep_28.md:3189:/bin/zsh -lc "pdftotext -f 1 -l 8 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '35,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:3299:   143	between person images and texts. Compared to conventional
./reviews/deep_28.md:3300:   144	image-based person retrieval (Re-ID) [2], [4]–[6] and attributebased person retrieval [7], text-based person retrieval [8] provides a more intuitive and convenient way by forming queries
./reviews/deep_28.md:3314:   158	trainable image-text pairs, CLIP contains abundant generic
./reviews/deep_28.md:3341:   185	visual objects and textual descriptions can serve as knowledge
./reviews/deep_28.md:3393:   237	demonstrates greater advantages in data-scarce scenarios especially in RSTPReid dataset. It achieves better performance
./reviews/deep_28.md:3398:   242	method for text-based person retrieval. To the best of our
./reviews/deep_28.md:3413:   257	In this section, we will briefly review the most relevant study including vision-language pre-training, parameterefficient transfer learning, and text-based person retrieval.
./reviews/deep_28.md:3419:前三篇的方法段已经能看出各自的“问题重写方式”：UP-Person 把全量微调改写成“知识保留不足”，HDNet 把跨模态差异拆成“相机、模态、身份冗余的层级污染”，MPT 把文本提示从静态辅助改写成“能和图像编码器互相强化的中间模态”。我继续补齐剩下两篇的方法段，并把消融和可视化也读出来，用来回答“证据闭环”。
./reviews/deep_28.md:3429:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '414,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:3467:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '68,200p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:3491:    90	There are three primary causes of noisy labels in this context:
./reviews/deep_28.md:3499:    98	imaging often loses key features such as color and texture,
./reviews/deep_28.md:3516:   115	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
./reviews/deep_28.md:3620:/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '436,616p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:3622:/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '33,160p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:3624:/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '161,292p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:3631:   166	trained for next-word prediction, generates text based on the encoded
./reviews/deep_28.md:3635:   170	instruction-following, semantic understanding, and text generation
./reviews/deep_28.md:3767:    40	from a single modality, cross-modal learning has received close attention recently. For example, in the context of the development of
./reviews/deep_28.md:3769:    42	on the representative VLM model CLIP [9] to leverage text semantics. As shown in Fig. 1 (b), it enhances visual features through
./reviews/deep_28.md:3770:    43	cross-modal contrastive learning with image-text pairs. Meanwhile,
./reviews/deep_28.md:3815:    88	attention due to their powerful capabilities in text generation and
./reviews/deep_28.md:3828:   101	the reasoning and understanding capabilities of LVLMs, while adhering to the mainstream ReID paradigm of feature extraction combined with feature similarity-based retrieval [1]. A potential solution involves using LVLMs to describe pedestrian images and finetuning the visual encoder via tasks such as image-text matching or
./reviews/deep_28.md:3830:   103	(1) High-quality and diverse text annotations are expensive to obtain.
./reviews/deep_28.md:3831:   104	(2) The goals of image-text matching or image caption prediction
./reviews/deep_28.md:4010:   556	a VLM pre-trained on large-scale image-text pairs, and it discards
./reviews/deep_28.md:4011:   557	the text encoder during inference. Differently, LVLM-ReID integrates LVLM into ReID training and inference stages in a novel
./reviews/deep_28.md:4073:/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '317,739p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:4500:有两篇排版把方法和表格混在同一页，文本顺序被列布局打乱了。我会按页和行号再补一次，重点补 HDNet 的跨模态重组末尾、LVLM-ReID 的训练和推理段，以及各篇消融证据。
./reviews/deep_28.md:4526:/bin/zsh -lc "pdftotext -f 9 -l 10 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:4790:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | rg -n \"Ablation|ablation|MPT|two-stage|prompts|visualization|t-SNE|alternating|knowledge propagation|alignment|mutual|baseline\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:4801:57:by two-stage training strategy [5]. They fix image encoders to optimize text prompts in the first stage and fix text prompts to optimize
./reviews/deep_28.md:4802:59:image encoders in the second stage. The generated text prompts
./reviews/deep_28.md:4803:64:However, the text prompts generated by multimodal large language models can not effectively describe identity information of
./reviews/deep_28.md:4804:65:visible-infrared person re-identification, text prompts, knowledge
./reviews/deep_28.md:4806:78:to improve the quality of text prompts.
./reviews/deep_28.md:4811:90:strategy which alternately optimizes the text prompts and image
./reviews/deep_28.md:4812:92:encoders in different epochs. When optimizing text prompts, we
./reviews/deep_28.md:4823:115:prompts (image encoders) and 𝐹𝑡𝑒𝑥𝑡 (𝐹𝑖𝑚𝑔 ) to represent the optimization process of text prompts (image encoders), the optimization process can be formalized as:
./reviews/deep_28.md:4829:186:optimizing text prompts and the modality alignment loss is used
./reviews/deep_28.md:4831:196:modality alignment loss. The loss treats the text features as a special intermediate modality, and aligns both visible and infrared
./reviews/deep_28.md:4832:197:modalities to the text modality. In this way, prompts can serve
./reviews/deep_28.md:4833:200:help of MPT, text prompts can help image encoders become better,
./reviews/deep_28.md:4835:203:both text prompts and image encoders will continue to improve.
./reviews/deep_28.md:4841:234:each other. Under such a premise, we propose the modality knowledge propagation loss. The loss constrains the predicted class probability of text features to be similar to that of image features. In
./reviews/deep_28.md:4842:235:this way, person-related knowledge is propagated from image encoders into text prompts. The loss function can be expressed as:
./reviews/deep_28.md:4844:240:to generate text prompts, but such strategy can only generate prompts
./reviews/deep_28.md:4848:247:structure of MPT is shown in the Fig. 2. Specifically, the text encoder takes prompts as input and outputs corresponding text features, while image encoders take images as input and output corresponding image features. When 𝑒𝑝𝑜𝑐ℎ = 2𝑖, we fix image encoders and optimize prompts with modality knowledge propagation loss. And when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 + 1, we fix prompts and optimize
./reviews/deep_28.md:4853:285:we can consider optimized prompts as a special intermediate modality. When optimizing encoders, we treat the text features generated by prompts as the “bridge” and use them to connect the two
./reviews/deep_28.md:4854:287:The proposed modality alignment loss first calculates the similarity between text features and image features, then utilizes contrastive learning loss to align the image features of both modalities
./reviews/deep_28.md:4862:545:number of text prompts. 𝑞𝑘 is the value in the target distribution,
./reviews/deep_28.md:4864:566:With the help of the above loss functions, text prompts and image encoders can mutually promote each other, ultimately improving the performance of cross-modality retrieval.
./reviews/deep_28.md:4872:639:two modalities to text features. This effectively reduces the discrepancies between the two modalities. More importantly, we utilize the knowledge distillation technique to help prompts learn
./reviews/deep_28.md:4889:672:The result means that text prompts and image encoders can indeed promote each other, ultimately enhancing the model’s ability
./reviews/deep_28.md:4891:674:𝐿𝑀𝑃 𝑇 ) improves 1.69% and 2.23% over two-stage training strategy. This indicates that by improving the quality of generated text
./reviews/deep_28.md:4898:691:image encoders and text prompts mutually promote each other. In
./reviews/deep_28.md:4899:692:this way, we obtain text prompts with more person-related information, which better align visible and infrared modalities at semantic level. In addition, we design the dedicated loss functions
./reviews/deep_28.md:4906:/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '253,435p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:5093:/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | rg -n \"Ablation|S-Prefix|L-Adapter|LoRA|Prefix|Adapter|component|conflict|parameters|Visualization|visualization|Full Tuning|full-tuning|Knowledge\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:5181:649:transformer block for both image and text encoders. In addition to prefix tokens in the keys and values of MHA, S-Prefix proposes a Sp factor in attention
./reviews/deep_28.md:5190:719:LoRA and Adapter within image and text encoders. (1)
./reviews/deep_28.md:5236:1174:rank of LoRA r is set to 32, while for RSTPReid, due to its
./reviews/deep_28.md:5238:1195:respectively while only fine-tuning a few parameters. Moreover, we demonstrate that full-tuning method IRRA has a
./reviews/deep_28.md:5253:1274:S-Prefix (63.69% R@1) improves attention by adding contextspecific embeddings, surpassing the prompt-based submodule
./reviews/deep_28.md:5280:1570:Fig. 6. The analysis of hyper-parameters at R@1 on CUHK-PEDES, ICFG-PEDES and RSTPReid.
./reviews/deep_28.md:5340:2227:and RSTPReid, respectively. Moreover, it is obvious that UPPerson is superior than IRRA-CLIP (full-tuning) in all metrics
./reviews/deep_28.md:5343:2234:V. Thus, full-tuning-based methods such as IRRA and IRRACLIP can cause the more severe overfitting when lacking of
./reviews/deep_28.md:5345:2248:infer that the full-tuning method IRRA, which only utilizes
./reviews/deep_28.md:5356:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | rg -n \"Ablation|ablation|CDM|MIDM|FCRS|Adversarial|Visualization|visualization|t-SNE|decoupling|camera|modality|recombination\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:5533:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | rg -n \"Ablation|ablation|RHL|Hybrid|GCE|clean|indistinguishable|obvious|real-world|SYSU-Real|noise|robust|visualization|distribution\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:5595:297:emerging studies have focused on label noise in text-to-image
./reviews/deep_28.md:5747:/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1260,1332p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:5762:  1273	textual descriptions and person image features. Specifically,
./reviews/deep_28.md:5763:  1274	S-Prefix (63.69% R@1) improves attention by adding contextspecific embeddings, surpassing the prompt-based submodule
./reviews/deep_28.md:5804:  1315	capacity, whereas RSTPReid, with its smaller dataset size,
./reviews/deep_28.md:5807:  1318	(10), ICFG-PEDES (14), and RSTPReid (2). This reflects
./reviews/deep_28.md:5828:/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1568,1820p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:5830:  1568	∗ indicates our replication results after a minor bug correction, also viewed as data augmentation in vanilla IRRA.
./reviews/deep_28.md:5832:  1570	Fig. 6. The analysis of hyper-parameters at R@1 on CUHK-PEDES, ICFG-PEDES and RSTPReid.
./reviews/deep_28.md:5861:  1599	APTM (w/o MALS) [71]
./reviews/deep_28.md:5862:  1600	APTM (w/ MALS) [71]
./reviews/deep_28.md:5911:  1649	IRRA-CLIP [13]
./reviews/deep_28.md:5912:  1650	IRRA∗ [13]
./reviews/deep_28.md:5913:  1651	IRRA [13]
./reviews/deep_28.md:5959:  1697	C OMPARISON ON RSTPR EID .
./reviews/deep_28.md:5966:  1704	APTM (w/o MALS) [71]
./reviews/deep_28.md:5967:  1705	APTM (w/ MALS) [71]
./reviews/deep_28.md:6003:  1741	IRRA-CLIP [13]
./reviews/deep_28.md:6004:  1742	IRRA∗ [13]
./reviews/deep_28.md:6005:  1743	IRRA [13]
./reviews/deep_28.md:6089:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '707,782p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6181:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | nl -ba | sed -n '613,688p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6209:   639	two modalities to text features. This effectively reduces the discrepancies between the two modalities. More importantly, we utilize the knowledge distillation technique to help prompts learn
./reviews/deep_28.md:6231:   661	feature maps of 𝑊𝐼 ⋅ 𝐼𝑖 to approximate that of 𝑊𝑇 ⋅ 𝑇𝑖 (text feature maps cannot be visualized). As shown in Fig. 5, feature maps
./reviews/deep_28.md:6242:   672	The result means that text prompts and image encoders can indeed promote each other, ultimately enhancing the model’s ability
./reviews/deep_28.md:6244:   674	𝐿𝑀𝑃 𝑇 ) improves 1.69% and 2.23% over two-stage training strategy. This indicates that by improving the quality of generated text
./reviews/deep_28.md:6252:   682	increase in training time. Meanwhile, since the text modality is
./reviews/deep_28.md:6261:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '947,1102p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6333:  1017	context of VI-ReID, wherein the discrimination of identity
./reviews/deep_28.md:6421:/bin/zsh -lc "pdftotext 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | rg -n \"PSTG|SGI|CSS|semantic token|learnable token|visual-only|Visualization|retrieval|inference time|ablation|Ablation\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6426:101:the reasoning and understanding capabilities of LVLMs, while adhering to the mainstream ReID paradigm of feature extraction combined with feature similarity-based retrieval [1]. A potential solution involves using LVLMs to describe pedestrian images and finetuning the visual encoder via tasks such as image-text matching or
./reviews/deep_28.md:6504:/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1810,1855p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6535:  1839	implicit reasoning) based on IRRA [13]. It shows that SDM
./reviews/deep_28.md:6550:  1854	ICFG-PEDES and RSTPReid. We train models on the dataset
./reviews/deep_28.md:6554:/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '2220,2258p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6561:  2225	advantage compared with IRRA when source domain is RSTPReid, which gains a significant improvement on R@1 by
./reviews/deep_28.md:6563:  2227	and RSTPReid, respectively. Moreover, it is obvious that UPPerson is superior than IRRA-CLIP (full-tuning) in all metrics
./reviews/deep_28.md:6568:  2232	scarce. We observe that the size of RSTPReid is much less
./reviews/deep_28.md:6570:  2234	V. Thus, full-tuning-based methods such as IRRA and IRRACLIP can cause the more severe overfitting when lacking of
./reviews/deep_28.md:6575:  2239	IRRA (the first row) and our proposed UP-Person (the second
./reviews/deep_28.md:6579:  2243	retrieval. It can be seen that UP-Person can retrieve the corresponding pedestrian images for a query text more accurately.
./reviews/deep_28.md:6580:  2244	For example, IRRA cannot correctly recognize the phrase bob
./reviews/deep_28.md:6584:  2248	infer that the full-tuning method IRRA, which only utilizes
./reviews/deep_28.md:6587:  2251	training. If the model of IRRA does not fully “see” the relevant
./reviews/deep_28.md:6597:/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1908,1942p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6636:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '920,947p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6668:/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '739,920p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_28.md:6859:作者注意到一个反直觉点：在文本行人检索里，全量微调 CLIP 不一定是最好的。全量微调虽然能适配训练集，但会破坏或遗忘 CLIP 原本的大规模视觉语言知识，而且每个数据集都要存一套大模型。相反，参数高效微调保留 CLIP 主干，可能同时利用“CLIP 里已有的行人相关知识”和“训练集学到的任务知识”。第二个观察更具体：把 Prefix、LoRA、Adapter 直接堆在一起会互相干扰，普通 Prefix 在 TPR 上收敛慢、效果差。
./reviews/deep_28.md:6868:关键证据不是单纯 SOTA 表，而是组件消融和冲突消融。单独的 LoRA、S-Prefix、L-Adapter 都比 zero-shot CLIP 强，三者组合在 CUHK-PEDES 上达到最高 R@1 和 mAP，说明“全局、局部、任务提示”确实互补。S-Prefix 对 vanilla Prefix 有明显提升，并且缩放因子过大时会梯度爆炸，证明他们不是随便加参数，而是在修 Prefix 梯度问题。L-Adapter 对比 vanilla Adapter、LN-tuning 和混合 Adapter，支持“普通 Adapter 会和其他 PETL 冲突，LayerNorm 旁边更适合调全局分布”的说法。可视化里 UP-Person 比全量微调的 IRRA 更能识别细粒度或少见描述，也服务于“全量微调丢失 CLIP 原知识”的叙事。
./reviews/deep_28.md:6893:作者抓住的是 CLIP-ReID 类方法里的一个弱点：两阶段生成的 text prompts 往往不关注行人身份细节。原因是原始 CLIP 对 VI-ReID 不熟，直接生成的提示更偏通用语义，不一定关心跨模态行人身份。另一个观察是，如果同时优化 prompts 和 image encoders，二者目标不同，会互相干扰。
./reviews/deep_28.md:6896:它把 prompt 从“给图像编码器用的静态文本辅助”改写成“可学习的中间模态”。关键词是 mutual promotion、mutual reinforcement、person-related prompts、text features as bridge、modality knowledge propagation、modality alignment、alternating training。旧问题是“怎么用文本帮助可见光和红外对齐”，新问题是“提示和图像编码器怎样互相变好”。
./reviews/deep_28.md:6899:机制很自然。既然 prompts 缺少身份信息，就让训练好的图像编码器把身份知识传给 prompts。作者用分类概率分布里的 dark knowledge，通过 KL 让文本特征的类别概率接近图像特征的类别概率。既然高质量 prompts 可以作为模态无关桥梁，就在优化图像编码器时，把可见光和红外图像特征都对齐到文本特征。既然同时优化会互相干扰，就交替训练：偶数 epoch 固定图像编码器训 prompts，奇数 epoch 固定 prompts 训图像编码器。这个设计和“互相强化”的叙事绑定很紧。
./reviews/deep_28.md:6902:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./reviews/deep_28.md:6905:这篇主要卖视角。模块上就是 KL 蒸馏、对齐损失、交替优化，但它把 prompt 的角色从“静态文本”翻成“被图像编码器教育、再反过来教育图像编码器的中间模态”。这比单纯说“引入文本提示”更像一个方法故事。reviewer 买账的原因是机制简单，问题清楚，证据能直接回答“prompt 是否真的更关注身份”。
./reviews/deep_28.md:6927:作者注意到 LVLM 很强，但直接用于 ReID 很别扭。让 LVLM 逐个判断 query 和 gallery 是否同人，成本和 token 数都不可接受。让 LVLM 生成 caption 再做 image-text matching 也有问题：高质量文本标注贵，caption 目标和 ReID 目标不一致，推理阶段通常没有真正利用 LVLM 的生成能力。CLIP-ReID 类方法虽然用了文本，但往往训练后丢掉文本编码器。
./reviews/deep_28.md:6930:它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。
./reviews/deep_28.md:6933:机制也比较自然。为了不逐对比对，就仍然保留 ReID 的特征提取加相似度检索范式。为了用 LVLM 的语义理解能力，就用指令让 Qwen2-VL 针对行人图像生成一个 `<REID>` 语义 token，要求关注年龄、性别、衣着、生物特征。为了让这个 token 不只是孤立文本输出，设计 SGI，把语义 token 和视觉 tokens 拼起来经过 4 层 Transformer 交互，最后用强化后的语义 token 作为身份表示。为了处理相机变化，又加 camera semantic supplementation，最好的是把相机 embedding 加到视觉 patch embedding 上。训练用 ID loss 和 triplet loss，LLM 冻结但允许梯度穿过它更新视觉模型和 SGI。
./reviews/deep_28.md:6949:作者注意到一个反直觉点：在文本行人检索里，全量微调 CLIP 不一定是最好的。全量微调虽然能适配训练集，但会破坏或遗忘 CLIP 原本的大规模视觉语言知识，而且每个数据集都要存一套大模型。相反，参数高效微调保留 CLIP 主干，可能同时利用“CLIP 里已有的行人相关知识”和“训练集学到的任务知识”。第二个观察更具体：把 Prefix、LoRA、Adapter 直接堆在一起会互相干扰，普通 Prefix 在 TPR 上收敛慢、效果差。
./reviews/deep_28.md:6958:关键证据不是单纯 SOTA 表，而是组件消融和冲突消融。单独的 LoRA、S-Prefix、L-Adapter 都比 zero-shot CLIP 强，三者组合在 CUHK-PEDES 上达到最高 R@1 和 mAP，说明“全局、局部、任务提示”确实互补。S-Prefix 对 vanilla Prefix 有明显提升，并且缩放因子过大时会梯度爆炸，证明他们不是随便加参数，而是在修 Prefix 梯度问题。L-Adapter 对比 vanilla Adapter、LN-tuning 和混合 Adapter，支持“普通 Adapter 会和其他 PETL 冲突，LayerNorm 旁边更适合调全局分布”的说法。可视化里 UP-Person 比全量微调的 IRRA 更能识别细粒度或少见描述，也服务于“全量微调丢失 CLIP 原知识”的叙事。
./reviews/deep_28.md:6983:作者抓住的是 CLIP-ReID 类方法里的一个弱点：两阶段生成的 text prompts 往往不关注行人身份细节。原因是原始 CLIP 对 VI-ReID 不熟，直接生成的提示更偏通用语义，不一定关心跨模态行人身份。另一个观察是，如果同时优化 prompts 和 image encoders，二者目标不同，会互相干扰。
./reviews/deep_28.md:6986:它把 prompt 从“给图像编码器用的静态文本辅助”改写成“可学习的中间模态”。关键词是 mutual promotion、mutual reinforcement、person-related prompts、text features as bridge、modality knowledge propagation、modality alignment、alternating training。旧问题是“怎么用文本帮助可见光和红外对齐”，新问题是“提示和图像编码器怎样互相变好”。
./reviews/deep_28.md:6989:机制很自然。既然 prompts 缺少身份信息，就让训练好的图像编码器把身份知识传给 prompts。作者用分类概率分布里的 dark knowledge，通过 KL 让文本特征的类别概率接近图像特征的类别概率。既然高质量 prompts 可以作为模态无关桥梁，就在优化图像编码器时，把可见光和红外图像特征都对齐到文本特征。既然同时优化会互相干扰，就交替训练：偶数 epoch 固定图像编码器训 prompts，奇数 epoch 固定 prompts 训图像编码器。这个设计和“互相强化”的叙事绑定很紧。
./reviews/deep_28.md:6992:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./reviews/deep_28.md:6995:这篇主要卖视角。模块上就是 KL 蒸馏、对齐损失、交替优化，但它把 prompt 的角色从“静态文本”翻成“被图像编码器教育、再反过来教育图像编码器的中间模态”。这比单纯说“引入文本提示”更像一个方法故事。reviewer 买账的原因是机制简单，问题清楚，证据能直接回答“prompt 是否真的更关注身份”。
./reviews/deep_28.md:7017:作者注意到 LVLM 很强，但直接用于 ReID 很别扭。让 LVLM 逐个判断 query 和 gallery 是否同人，成本和 token 数都不可接受。让 LVLM 生成 caption 再做 image-text matching 也有问题：高质量文本标注贵，caption 目标和 ReID 目标不一致，推理阶段通常没有真正利用 LVLM 的生成能力。CLIP-ReID 类方法虽然用了文本，但往往训练后丢掉文本编码器。
./reviews/deep_28.md:7020:它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。
./reviews/deep_28.md:7023:机制也比较自然。为了不逐对比对，就仍然保留 ReID 的特征提取加相似度检索范式。为了用 LVLM 的语义理解能力，就用指令让 Qwen2-VL 针对行人图像生成一个 `<REID>` 语义 token，要求关注年龄、性别、衣着、生物特征。为了让这个 token 不只是孤立文本输出，设计 SGI，把语义 token 和视觉 tokens 拼起来经过 4 层 Transformer 交互，最后用强化后的语义 token 作为身份表示。为了处理相机变化，又加 camera semantic supplementation，最好的是把相机 embedding 加到视觉 patch embedding 上。训练用 ID loss 和 triplet loss，LLM 冻结但允许梯度穿过它更新视觉模型和 SGI。
./analyses/deep_11.txt:35:4. 证据闭环：关键证据是 clustering ablation 和 cluster number visualization。文本说明 MIE 带来约 10 到 15 个点 mAP，DNC 再带来 2 到 5 个点，合起来 14 到 17 个点。可视化里 visible cluster 数量更接近真实类别数，t-SNE 显示 MIE+DNC 后同身份跨模态更紧。objective ablation 里 inter-modality cluster loss 贡献最大，说明改好的跨模态伪标签确实在驱动训练。
./reviews/deep_8.md:14:你是 ReID 创新方法论拆解员 8。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_8.md:36:我会先按项目规则尝试读取目标文件，再用 `pdftotext` 抽取每篇前 8 页的方法部分。随后只在判断先例和视角新意需要时联网核对，最后按你给的 5 个问题逐篇拆。
./reviews/deep_8.md:52:/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:59:# PDF Skill
./reviews/deep_8.md:73:3. Use `pdfplumber` or `pypdf` for text extraction and quick checks; do not rely on text extraction for layout fidelity.
./reviews/deep_8.md:128:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reviews/deep_8.md:146:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_8.md:147:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_8.md:157:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_8.md:164:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_8.md:177:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_8.md:181:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_8.md:225:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_8.md:232:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_8.md:275:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_8.md:288:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_8.md:330:/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:332:/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:342:that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
./reviews/deep_8.md:343:between the textual and visual modalities within a unified latent representation space. To address this
./reviews/deep_8.md:398:identify the most relevant person image within a large gallery [31]. Unlike general image-text
./reviews/deep_8.md:399:retrieval [5, 6, 44, 47, 50], which tends to achieve semantic-based matching between text and image,
./reviews/deep_8.md:401:fine-grained correspondences between textual and visual modalities, owing to the large intra-class
./reviews/deep_8.md:404:viewpoints, and illumination conditions, and (2) textual descriptions are influenced by differences
./reviews/deep_8.md:405:in phrasing, word order, and textual ambiguities. Therefore, the primary challenges in TIPR are
./reviews/deep_8.md:406:how to extract discriminative global representations from image-text pairs and how to achieve
./reviews/deep_8.md:410:images and texts, which are projected into a joint embedding space. Their widely adopted loss
./reviews/deep_8.md:412:scalar projections of image-text pairs and their matched label indicators. In comparison, the SDM
./reviews/deep_8.md:414:between the normalized similarity profile of image-text pairs and the true label distribution. In
./reviews/deep_8.md:423:mechanisms to implicitly aggregate local image-text representations. As a result, they yield only the
./reviews/deep_8.md:425:multimodal representations correctly encode the corresponding visual and textual information.
./reviews/deep_8.md:427:(FMFA) framework that enhances global matching by including explicit fine-grained image-text
./reviews/deep_8.md:429:Distribution Matching (A-SDM) module to ensure the correct matching of positive image-text
./reviews/deep_8.md:444:the corresponding query text.
./reviews/deep_8.md:448:with a sparse similarity matrix. During this process, the sparse similarity matrix between text
./reviews/deep_8.md:449:and image reflects the contribution of textual and visual representations to the final multimodal
./reviews/deep_8.md:452:and its original visual and textual representations. These designs allow EFA to realize fine-grained
./reviews/deep_8.md:454:image-text representations without introducing additional supervision. FMFA is evaluated on three
./reviews/deep_8.md:459:— We present an A-SDM module aimed at precisely aligning image-text pairs in a shared
./reviews/deep_8.md:469:space that enables coherent alignment between visual and textual representations. Existing methods
./reviews/deep_8.md:471:Early global methods [59, 60] directly aligned the global representations of images and text in
./reviews/deep_8.md:474:loss to minimize the discrepancy between the scalar projection of image-text pairs and their labels.
./reviews/deep_8.md:477:methods [12, 42, 49] explicitly aligned local visual and textual features to achieve fine-grained
./reviews/deep_8.md:496:image-text pairs in the person Re-IDentification (ReID) domain to VLMs. Zuo et al. [61] utilized
./reviews/deep_8.md:498:textual descriptions for pedestrian images. Yang et al. [55] employed BLIP-2 [24] to produce
./reviews/deep_8.md:507:and text words, encouraging the model to infer missing semantics and implicitly learn local visualtextual correspondences. He et al. [17] proposed the Vision-Guided Semantic-Group (VGSG)
./reviews/deep_8.md:508:network to cluster textual tokens into semantic groups and align them with corresponding visual
./reviews/deep_8.md:510:a global representation space. Similarly, Jiang and Ye [21] developed IRRA to employ an Implicit
./reviews/deep_8.md:527:Motivated by the success of IRRA [21], we use the modified full CLIP [37] visual and textual
./reviews/deep_8.md:552:𝑣 is linearly mapped into the joint image-text embedding
./reviews/deep_8.md:555:Textual Modality. Given an input text 𝑇 , we utilize the CLIP-Xformer textual extractor [37] to
./reviews/deep_8.md:556:obtain its embedding. The text is first tokenized through lower-cased Byte Pair Encoding [40] and
./reviews/deep_8.md:564:linearly mapped into the joint image-text representation space, generating a compressed global
./reviews/deep_8.md:565:textual representation.
./reviews/deep_8.md:571:Adopted from IRRA [21], we introduce a novel A-SDM module, which aims to adaptively pull the
./reviews/deep_8.md:572:unmatched positive image-text pairs into a shared representation space, further enhancing the
./reviews/deep_8.md:580:Let the mini-batch contain 𝐵 image-text pairs, we pair each text embedding 𝑔𝑖𝑡 with its global
./reviews/deep_8.md:584:image-text pairs is obtained through the following softmax function:
./reviews/deep_8.md:597:The probability 𝑝𝑖,𝑗 quantifies how much the similarity between the text embedding 𝑔𝑖𝑡 and the
./reviews/deep_8.md:600:Let the 𝑖th text 𝑇𝑖 from the batch be designated as the query text and 𝐼𝑖 be the corresponding
./reviews/deep_8.md:601:image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
./reviews/deep_8.md:602:weighting factor by assessing the similarity between the query text 𝑇𝑖 and all image representations:
./reviews/deep_8.md:610:where 𝛼 is a weight factor reflecting the contribution of unmatched image-text pairs to the crossmodal global matching ability of the model. Here, max𝑘 𝑝𝑖,𝑘 indicates the top similarity value
./reviews/deep_8.md:611:between the text 𝑇𝑖 and every image within the mini-batch, while 𝑝𝑖,𝑖 refers to the similarity
./reviews/deep_8.md:617:global cross-modal alignment. The A-SDM loss for mapping text to image within a mini-batch is
./reviews/deep_8.md:636:In a complementary manner, the A-SDM loss for the image-to-text branch L𝑖2𝑡 is derived by
./reviews/deep_8.md:637:swapping the roles of the text and image features. The bidirectional A-SDM loss is formulated as:
./reviews/deep_8.md:646:between visual and textual modalities. Although many attention-based fine-grained alignment
./reviews/deep_8.md:647:approaches have shown effectiveness by implicitly associating local regions in images with textual
./reviews/deep_8.md:650:sparse similarity matrix between the local image and text features. To further reduce redundant
./reviews/deep_8.md:652:coding to align the aggregated language-grouped vision embeddings with both image and text
./reviews/deep_8.md:668:text token, which limits scalability to large batch sizes. Therefore, we apply a sparsification strategy
./reviews/deep_8.md:673:An image 𝐼 and its corresponding text 𝑇 are encoded through the visual and textual encoders,
./reviews/deep_8.md:674:respectively. As presented in Figure 3(a), the similarity between image patches and text tokens is
./reviews/deep_8.md:675:𝐿 of the text transformer and {𝑓 𝑣 } 𝑁
./reviews/deep_8.md:680:of the ViT. 𝑠𝑖,𝑗 = 𝑓𝑖 · 𝑓 𝑗 measures the similarity between the text token 𝑓𝑖𝑡 and the image patch 𝑓 𝑗𝑣 ,
./reviews/deep_8.md:736:𝐿 as the text token 𝑓𝑖𝑡 .
./reviews/deep_8.md:739:𝐿 as well as image embeddings {𝑓 𝑣 } 𝑁 , retheir corresponding original text embeddings {𝑓𝑖𝑡 }𝑖=1
./reviews/deep_8.md:742:computation between the joint embeddings and their corresponding text and image embeddings,
./reviews/deep_8.md:744:present the calculation between the joint embeddings and the text embeddings, while the remaining
./reviews/deep_8.md:746:For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
./reviews/deep_8.md:750:𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
./reviews/deep_8.md:768:Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
./reviews/deep_8.md:790:where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
./reviews/deep_8.md:792:Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
./reviews/deep_8.md:794:in Figure 3(b). We calculate the EFA loss from the text to its joint embedding, adapted from the
./reviews/deep_8.md:884:Virtualtext
./reviews/deep_8.md:886:Virtual-text
./reviews/deep_8.md:963:Figure 1: (1) Examples of different types of image-text pairs.
./reviews/deep_8.md:964:Identity-bounded Matching (IBM) loss is designed to handle different types of image-text pairs distinctly by establishing different
./reviews/deep_8.md:965:similarity boundaries, leading to more accurate cross-modal correspondences. (2) We propose a cross-modal collaborative representation learning framework that leverages the inter-modality complementarity to enhance image features without relying on actual texts.
./reviews/deep_8.md:983:text description. Current TIPR methods mainly
./reviews/deep_8.md:985:and texts, ignoring their potential complementarity. Besides, existing matching losses treat all positive pairs from the same identity equally, leading to noisy correspondences. In this paper, we
./reviews/deep_8.md:987:TIPR by effectively leveraging the complementarity between modalities. The text typically contains
./reviews/deep_8.md:990:information. However, corresponding text for each
./reviews/deep_8.md:991:image is not always available, especially during inference. Accordingly, we introduce a Virtual-text
./reviews/deep_8.md:993:virtual-text features for cross-modal collaboration,
./reviews/deep_8.md:994:eliminating the need for actual texts. We then design a Cross-Modal Collaboration learning process,
./reviews/deep_8.md:996:loss to promote interaction and fusion between image and virtual-text features for mutual enhancement. Additionally, an Identity-bounded Matching
./reviews/deep_8.md:997:loss is proposed to handle different types of imagetext pairs distinctly, leading to more accurate crossmodal correspondences. Extensive experiments on
./reviews/deep_8.md:1014:certain cameras. As a realistic extension to ReID, TIPR assumes that text descriptions are available for the missing images, allowing the retrieval of target person images via these
./reviews/deep_8.md:1016:texts, making TIPR a challenging and meaningful task.
./reviews/deep_8.md:1027:between images and texts, often overlooking their potential
./reviews/deep_8.md:1028:complementarity. In reality, the text primarily contains pedestrian identity details with less noise. Even when pedestrians
./reviews/deep_8.md:1029:are visually similar, their text descriptions are often distinct,
./reviews/deep_8.md:1030:making text information more discriminative and robust. This
./reviews/deep_8.md:1031:leads us to explore whether inter-modality complementary information can benefit the TIPR task. Inspired by this, we propose integrating text information into images to emphasize
./reviews/deep_8.md:1034:access to the corresponding text for each image, which is not
./reviews/deep_8.md:1039:image features without relying on actual texts. Specifically,
./reviews/deep_8.md:1040:we introduce a Virtual-text Embedding Synthesizer (VES),
./reviews/deep_8.md:1042:to produce high-quality virtual-text embeddings directly from
./reviews/deep_8.md:1043:images, eliminating the need for actual texts. These generated
./reviews/deep_8.md:1044:virtual-text embeddings can replace actual texts in subsequent
./reviews/deep_8.md:1046:the generated virtual-text embeddings with actual text embeddings at both the feature and semantic levels. To fully exploit
./reviews/deep_8.md:1047:the complementarity between images and texts, we design a
./reviews/deep_8.md:1049:The visual backbone branch focuses on generating image features. The virtual-text branch uses VES to create virtual-text
./reviews/deep_8.md:1051:to produce virtual-text features adapted to the target domain.
./reviews/deep_8.md:1052:The CRCL loss enforces that image and virtual-text features
./reviews/deep_8.md:1053:maintain the same relationship with modality-specific prototypes. This encourages information exchange and collaboration between the two branches, allowing virtual textual information to emphasize the discriminative and modality-shared
./reviews/deep_8.md:1059:contains three types of image-text pairs: single-view strong
./reviews/deep_8.md:1063:three types of image-text pairs should decrease progressively.
./reviews/deep_8.md:1069:each category of image-text pairs, thereby leading to more
./reviews/deep_8.md:1074:complementary information to improve the TIPR task without relying on actual texts. (2) Identity-bounded matching loss is proposed to learn precise cross-modal correspondences. (3) Extensive experiments verify the effectiveness
./reviews/deep_8.md:1100:been proposed to align images and texts. Early methods [Li et al., 2017; Zhang and Lu, 2018] directly aligned
./reviews/deep_8.md:1101:the global features of images and texts. To achieve finegrained correspondences, later methods [Chen et al., 2022;
./reviews/deep_8.md:1107:Despite these advancements, existing methods mainly emphasize image-text associations but overlook their complementarity. In this study, we explore inter-modality complementarity to enhance image features with text information.
./reviews/deep_8.md:1108:Besides, existing matching losses treat image-text pairs of the
./reviews/deep_8.md:1113:boundaries for different types of image-text pairs.
./reviews/deep_8.md:1121:Stage 1: Virtual-text Embedding
./reviews/deep_8.md:1163:Virtual-text features
./reviews/deep_8.md:1165:Virtual-text Branch
./reviews/deep_8.md:1193:Visual/Virtualtext Features
./reviews/deep_8.md:1238:Figure 2: Overview of our CoRL. In the first stage, we introduce a Virtual-text Embedding Synthesizer to generate a virtual-text embedding
./reviews/deep_8.md:1239:from each image. The generated virtual-text embedding is then used to replace actual text in modeling multi-modal complementarity with
./reviews/deep_8.md:1241:Consistency Loss, which encourages the exchange of information and mutual enhancement between image and virtual-text features.
./reviews/deep_8.md:1248:models on vast collections of image-text pairs, demonstrating exceptional capabilities in semantic understanding [Shen
./reviews/deep_8.md:1251:achieve a deep understanding of both image and text semantics and their correspondences. Some studies [Li et al.,
./reviews/deep_8.md:1253:texts with tasks such as mask language modeling and image captioning. However, these methods typically necessitate pairwise interactions for all image-text pairs, leading
./reviews/deep_8.md:1257:contrastive representation learning into Vision-language pretraining. These methods encode images and texts separately
./reviews/deep_8.md:1260:image-text pairs. The strong multi-modal alignment capability of CLIP prompts us to introduce it into the TIPR task.
./reviews/deep_8.md:1273:The TIPR dataset contains image-text pairs of persons with
./reviews/deep_8.md:1275:collected from different cameras with their annotated text descriptions. The goal is to accurately identify images that belong to the same identity as a given text description. This requires the TIPR model to extract identity-discriminative representations and accurately establish cross-modal correspondences. To achieve this, we first use a pre-trained CLIP as
./reviews/deep_8.md:1290:Current TIPR methods mainly emphasize image-text associations but overlook their complementarity. Texts offer
./reviews/deep_8.md:1291:the following advantages over images: texts primarily contain pedestrian identity details with less noise. Even when
./reviews/deep_8.md:1292:pedestrians are visually similar, their text descriptions are
./reviews/deep_8.md:1293:often distinct, making text information more discriminative
./reviews/deep_8.md:1294:and robust. Complementing images with textual information
./reviews/deep_8.md:1296:gaps. However, this requires access to corresponding text for
./reviews/deep_8.md:1298:eliminating the dependency on actual text and effectively utilizing text information to enhance image representations. To
./reviews/deep_8.md:1300:develop a Virtual-text Embedding Synthesizer (VES) to generate virtual-text embeddings directly from each image, eliminating the need for actual paired text. In the second stage, we
./reviews/deep_8.md:1302:process to integrate the virtual-text information into images,
./reviews/deep_8.md:1304:Virtual-text Embedding Synthesizer. To eliminate reliance on actual paired text, we propose a Virtual-text Embedding Synthesizer (VES) that generates a virtual-text embedding directly from each image, as illustrated in Figure 2.
./reviews/deep_8.md:1305:CLIP, trained on vast amounts of image-text data, can generate modality-aligned image-text features. We utilize its
./reviews/deep_8.md:1306:alignment capabilities to convert images into text embeddings. VES functions as a CLIP-based encoder-decoder
./reviews/deep_8.md:1312:This embedding is then transformed into virtual-text tokens
./reviews/deep_8.md:1313:through a fully connected layer. Finally, CLIP’s textual encoder Et decodes these virtual-text tokens to generate the
./reviews/deep_8.md:1314:virtual-text embedding ẑt,i .
./reviews/deep_8.md:1315:This generated virtual-text embedding ẑt,i can serve as a
./reviews/deep_8.md:1316:substitute for the actual text embedding to enhance its image
./reviews/deep_8.md:1317:feature, given that the virtual-text and actual-text embeddings
./reviews/deep_8.md:1320:align that aligns the generated virtual-text
./reviews/deep_8.md:1321:embedding with the actual text embedding zt,i at both feature
./reviews/deep_8.md:1323:text Ti .
./reviews/deep_8.md:1368:where zt,i denotes the actual text embedding of text Ti , generated by CLIP’s visual encoder. ∥ · ∥2 denotes the L2 distance,
./reviews/deep_8.md:1370:conversion between images and virtual-text embeddings, we
./reviews/deep_8.md:1436:the virtual-text feature t̂i , we introduce a cross-modal relation consistency loss (CRCL). This loss enforces that the image and virtual-text features share the same relationship with
./reviews/deep_8.md:1439:each batch B, we cluster the image and text features based
./reviews/deep_8.md:1440:on identity labels to obtain visual and textual prototypes, pv
./reviews/deep_8.md:1444:rit between the virtual-text feature t̂i and these textual prototypes pt . Thus, the CRCL is formulated as:
./reviews/deep_8.md:1458:knowledge distillation on instance-to-prototype relations between the visual embedding vi and virtual-text embedding t̂i
./reviews/deep_8.md:1459:allows them to exchange information and benefit from one another. Incorporating virtual-text information emphasizes the
./reviews/deep_8.md:1461:between the image Ii and the text Ti .
./reviews/deep_8.md:1470:VES enables the generation of virtual-text embedding that is
./reviews/deep_8.md:1471:aligned with the actual text for each image, eliminating the
./reviews/deep_8.md:1472:need for actual text.
./reviews/deep_8.md:1473:Cross-modal Collaboration. With the virtual-text embedding, we design a dual-branch cross-modal collaboration
./reviews/deep_8.md:1475:image-text pair (Ii , Ti ), the visual backbone branch generates
./reviews/deep_8.md:1476:the image feature vi of image Ii . Simultaneously, the virtualtext branch first produces a virtual-text embedding ẑt,i from
./reviews/deep_8.md:1478:g(·) to yield a virtual-text feature t̂i adapted to the target domain. Additionally, the text feature ti for text Ti is generated
./reviews/deep_8.md:1479:via the textual backbone. We optimize the Adapter by aligning the virtual-text feature t̂i with the text feature ti through
./reviews/deep_8.md:1490:When a batch contains multiple image-text pairs of the same
./reviews/deep_8.md:1492:all images and texts in the batch. Due to view variations,
./reviews/deep_8.md:1494:can lead to noisy matching of cross-view image-text pairs of
./reviews/deep_8.md:1496:image and its corresponding text form a single-view strong
./reviews/deep_8.md:1497:positive pair, while image-text pairs across different views
./reviews/deep_8.md:1498:are considered as cross-view weak positive pairs. Imagetext pairs with different identities are categorized as negative
./reviews/deep_8.md:1508:with the corresponding text. Thus, each batch contains B
./reviews/deep_8.md:1509:= P K images and their corresponding texts. This results in
./reviews/deep_8.md:1513:similarity for these image-text pairs, denoted as {ssp
./reviews/deep_8.md:1522:guish different image-text pairs, we propose an IBM loss that
./reviews/deep_8.md:1590:we compute the cross entropy loss Lid on image and text features to classify them by identity. This loss forces the network
./reviews/deep_8.md:1599:stage, we train a virtual-text embedding synthesizer, where
./reviews/deep_8.md:1600:CLIP’s visual and textual encoders are frozen and only a fully
./reviews/deep_8.md:1602:V ES to convert image embeddings to virtual-text tokens. We pre-extract and save features from the dataset, allowing us to focus on optimizing this
./reviews/deep_8.md:1609:During inference, both visual and virtual-text features of
./reviews/deep_8.md:1611:to the query text features. The final retrieval score is obtained
./reviews/deep_8.md:1625:2021] consists of 54,522 image-text pairs of 4,102 persons,
./reviews/deep_8.md:1627:pairs from 3,102 people, with the remaining 1,000 people reserved for evaluation. RSTPReid [Zhu et al., 2021] includes
./reviews/deep_8.md:1633:the text sequence is set to 77, and random masking is employed for text augmentation. We use CLIP-ViT-B/16 as the
./reviews/deep_8.md:1638:fine-tune the visual/textual backbones with an initial learning
./reviews/deep_8.md:1655:ICFG-PEDES, and RSTPReid. Our CoRL achieves leading performance on these benchmarks, underscoring its effectiveness and advantages. On the CUHK-PEDES dataset,
./reviews/deep_8.md:1660:the RSTPReid dataset, CoRL delivers impressive results with
./reviews/deep_8.md:1665:cross-modal complementarity and establishing precise crossmodal correspondences. CoRL effectively integrates textual information into images to enhance their discriminability
./reviews/deep_8.md:1667:matching loss to address noisy correspondences among different types of image-text pairs.
./reviews/deep_8.md:1685:IRRA [Jiang and Ye, 2023]
./reviews/deep_8.md:1700:RDE [Qin et al., 2024]
./reviews/deep_8.md:1702:APTM [Yang et al., 2023]
./reviews/deep_8.md:1850:RSTPReid
./reviews/deep_8.md:2027:loss treats image-text pairs of the same identity equally, leading to noisy correspondences and poor performance. Comparing 1# (6#) and 0# (5#), our IBM loss creates distinct
./reviews/deep_8.md:2028:boundaries for differentiating various categories of imagetext pairs and establishes more precise cross-modal correspondences, achieving a 4.03% (2.30%) R@1 improvement.
./reviews/deep_8.md:2029:Comparing 2# (7#) and 0# (5#), VES generates a virtual-text
./reviews/deep_8.md:2031:facilitates information exchange between image and virtualtext features, allowing them to benefit from each other and
./reviews/deep_8.md:2043:and actual text embeddings at both feature and semantic levels, which is crucial for ensuring that VES generates virtualtext embeddings aligned with actual text. (2) The introduction of Ls1
./reviews/deep_8.md:2047:high-quality virtual-text embeddings.
./reviews/deep_8.md:2143:Computational Complexity: We compare the computational cost and inference time of CoRL with classic methods in Table 5. TIPCB and CFine build fine-grained correspondences by learning local features, reducing retrieval efficiency due to pairwise similarity calculations. IRRA and
./reviews/deep_8.md:2146:reducing parameters. The virtual-text feature in CoRL adds
./reviews/deep_8.md:2148:Qualitative Results: We qualitatively assess the effectiveness of our CoRL in Figure 4, showcasing the Top-10 retrieved images for each query text using both Baseline and
./reviews/deep_8.md:2191:IRRA [Jiang and Ye, 2023]
./reviews/deep_8.md:2221:ages with the same identity as the given query text are ranked
./reviews/deep_8.md:2230:In this paper, we propose leveraging multi-modal complementarity to improve the TIPR task. Specifically, we introduce a virtual-text embedding synthesizer that generates
./reviews/deep_8.md:2231:high-quality virtual-text embeddings aligned with actual text
./reviews/deep_8.md:2232:from images. These virtual-text embeddings replace actual
./reviews/deep_8.md:2233:texts, allowing us to explore multimodal complementarity
./reviews/deep_8.md:2234:with images and eliminating the dependency on actual texts.
./reviews/deep_8.md:2236:consistency loss, which enforces the exchange of information and mutual enhancement between image and virtual-text
./reviews/deep_8.md:2239:of image-text pairs, establishing more accurate cross-modal
./reviews/deep_8.md:2260:Zhang. Rasa: Relation and sensitivity aware representation learning for text-based person search. In International
./reviews/deep_8.md:2263:and Min Zhang. An empirical study of clip for text-based
./reviews/deep_8.md:2267:text-based person search. Neurocomputing, 494:171–181,
./reviews/deep_8.md:2273:Shao, and Dacheng Tao. Semantically self-aligned network for text-to-image part-aware person re-identification,
./reviews/deep_8.md:2291:Jiang, and Henghui Ding. Vgsg: Vision-guided semanticgroup network for text-based person search. IEEE Transactions on Image Processing, 2023.
./reviews/deep_8.md:2293:implicit relation reasoning and aligning for text-to-image
./reviews/deep_8.md:2318:cross-modal evidential learning for text-based person retrieval. In ACM International Conference on Multimedia,
./reviews/deep_8.md:2324:Tao, and Xi Luo. Learning semantic polymorphic mapping for text-based person retrieval. IEEE Transactions on
./reviews/deep_8.md:2329:uncertainty-based learning for text-based person retrieval.
./reviews/deep_8.md:2332:representation learning for text-based person retrieval. In
./reviews/deep_8.md:2335:Mind the inconsistent semantics in positive pairs: Semantic aligning and multimodal contrastive learning for textbased pedestrian search. IEEE Transactions on Information Forensics and Security, 19:6409–6424, 2024.
./reviews/deep_8.md:2337:Xi Peng, Joey Tianyi Zhou, and Peng Hu. Noisycorrespondence learning for text-to-image person reidentification. In IEEE Conference on Computer Vision
./reviews/deep_8.md:2344:/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:2357:that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
./reviews/deep_8.md:2358:between the textual and visual modalities within a unified latent representation space. To address this
./reviews/deep_8.md:2405:identify the most relevant person image within a large gallery [31]. Unlike general image-text
./reviews/deep_8.md:2406:retrieval [5, 6, 44, 47, 50], which tends to achieve semantic-based matching between text and image,
./reviews/deep_8.md:2408:fine-grained correspondences between textual and visual modalities, owing to the large intra-class
./reviews/deep_8.md:2411:viewpoints, and illumination conditions, and (2) textual descriptions are influenced by differences
./reviews/deep_8.md:2412:in phrasing, word order and textual ambiguities. Therefore, the primary challenges in TIPR are
./reviews/deep_8.md:2413:how to extract discriminative global representations from image-text pairs and how to achieve
./reviews/deep_8.md:2417:images and texts, which are projected into a joint embedding space. Their widely adopted loss
./reviews/deep_8.md:2420:projections of image-text pairs and their matched label indicators. In comparison, the SDM loss
./reviews/deep_8.md:2422:the normalized similarity profile of image-text pairs and the true label distribution. In addition,
./reviews/deep_8.md:2432:attention mechanisms to implicitly aggregate local image-text representations. As a result, they
./reviews/deep_8.md:2435:the aggregated multimodal representations correctly encode the corresponding visual and textual
./reviews/deep_8.md:2438:including explicit fine-grained image-text alignment and existing implicit relational reasoning.
./reviews/deep_8.md:2440:ensure the correct matching of positive image-text pairs. Within the joint embedding space, the
./reviews/deep_8.md:2446:process, the sparse similarity matrix between text and image reflects the contribution of textual and
./reviews/deep_8.md:2455:Query text
./reviews/deep_8.md:2456:Query text
./reviews/deep_8.md:2477:Fig. 1. Evolution of global matching methods for text-to-image person retrieval. (a) Existing global matching
./reviews/deep_8.md:2481:them closer to the corresponding query text.
./reviews/deep_8.md:2484:the aggregated multimodal representation and its original visual and textual representations. These
./reviews/deep_8.md:2486:in learning more distinctive global image-text representations without introducing additional
./reviews/deep_8.md:2493:image-text pairs in a shared embedding space. It adaptively adjusts to narrow the distance
./reviews/deep_8.md:2505:textual representations. Existing methods can be typically classified into global and local matching
./reviews/deep_8.md:2507:Early global methods [59, 60] directly aligned the global representations of images and text in
./reviews/deep_8.md:2510:to minimize the discrepancy between the scalar projection of image-text pairs and their labels.
./reviews/deep_8.md:2513:methods [12, 42, 49] explicitly aligned local visual and textual features to achieve fine-grained
./reviews/deep_8.md:2528:image-text pairs in the person Re-IDentification (ReID) domain to VLMs. Zuo et al. [61] utilized
./reviews/deep_8.md:2530:textual descriptions for pedestrian images. Yang et al. [55] employed BLIP-2 [24] to produce
./reviews/deep_8.md:2540:and text words, encouraging the model to infer missing semantics and implicitly learn local
./reviews/deep_8.md:2541:visual-textual correspondences. He et al. [17] proposed the Vision-Guided Semantic-Group (VGSG)
./reviews/deep_8.md:2542:network to cluster textual tokens into semantic groups and align them with corresponding visual
./reviews/deep_8.md:2544:a global representation space. Similarly, Jiang et al. [21] developed IRRA to employ an Implicit
./reviews/deep_8.md:2562:Motivated by the success of IRRA [21], we use the modified full CLIP [37] visual and textual
./reviews/deep_8.md:2650:𝑣 is linearly mapped into the joint image-text
./reviews/deep_8.md:2653:Textual Modality. Given an input text 𝑇 , we utilize the CLIP-Xformer textual extractor [37] to
./reviews/deep_8.md:2654:obtain its embedding. The text is first tokenized through lower-cased Byte Pair Encoding (BPE)
./reviews/deep_8.md:2661:𝑡 , is linearly mapped into the joint image-text representation space, generating a compressed
./reviews/deep_8.md:2663:global textual representation.
./reviews/deep_8.md:2675:Adopted from IRRA [21], we introduce a novel Adaptive Similarity Distribution Matching (A-SDM)
./reviews/deep_8.md:2676:module, which aims to adaptively pull the unmatched positive image-text pairs into a shared
./reviews/deep_8.md:2678:Let the mini-batch contain 𝐵 image-text pairs, we pair each text embedding 𝑔𝑖𝑡 with its global
./reviews/deep_8.md:2682:matrix of image-text pairs is obtained through the following softmax function:
./reviews/deep_8.md:2692:probability 𝑝𝑖,𝑗 quantifies how much the similarity between the text embedding 𝑔𝑖𝑡 and the image
./reviews/deep_8.md:2695:Let the 𝑖 − 𝑡ℎ text 𝑇𝑖 from the batch be designated as the query text and 𝐼𝑖 be the corresponding
./reviews/deep_8.md:2696:image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
./reviews/deep_8.md:2697:weighting factor by assessing the similarity between the query text 𝑇𝑖 and all image representations:
./reviews/deep_8.md:2705:where 𝛼 is a weight factor reflecting the contribution of unmatched image-text pairs to the crossmodal global matching ability of the model. Here, max𝑘 𝑝𝑖,𝑘 indicates the top similarity value
./reviews/deep_8.md:2706:between the text 𝑇𝑖 and every image within the mini-batch, while 𝑝𝑖,𝑖 refers to the similarity
./reviews/deep_8.md:2712:global cross-modal alignment. The A-SDM loss for mapping text to image within a mini-batch is
./reviews/deep_8.md:2733:In a complementary manner, the A-SDM loss for the image-to-text branch L𝑖2𝑡 is derived by
./reviews/deep_8.md:2734:swapping the roles of the text and image features. The bi-directional A-SDM loss is formulated as:
./reviews/deep_8.md:2743:between visual and textual modalities. Although many attention-based fine-grained alignment
./reviews/deep_8.md:2744:approaches have shown effectiveness by implicitly associating local regions in images with textual
./reviews/deep_8.md:2793:sparse similarity matrix between the local image and text features. To further reduce redundant
./reviews/deep_8.md:2795:coding to align the aggregated language-grouped vision embeddings with both image and text
./reviews/deep_8.md:2799:every text token, which limits scalability to large batch sizes. Therefore, we apply a sparsification
./reviews/deep_8.md:2804:An image 𝐼 and its corresponding text 𝑇 are encoded through the visual and textual encoders,
./reviews/deep_8.md:2805:respectively. As presented in Figure 3a, the similarity between image patches and text tokens is
./reviews/deep_8.md:2806:𝐿 of the text transformer and {𝑓 𝑣 } 𝑁
./reviews/deep_8.md:2809:of the vision transformer. 𝑠𝑖,𝑗 = 𝑓𝑖𝑡 · 𝑓 𝑗𝑣 measures the similarity between the text token 𝑓𝑖𝑡 and the
./reviews/deep_8.md:2866:𝐿 as the text token 𝑓𝑖𝑡 .
./reviews/deep_8.md:2873:and their corresponding original text embeddings {𝑓𝑖 }𝑖=1 as well as image embeddings {𝑓𝑖 }𝑖=1 ,
./reviews/deep_8.md:2875:computation between the joint embeddings and their corresponding text and image embeddings,
./reviews/deep_8.md:2877:present the calculation between the joint embeddings and the text embeddings, while the remaining
./reviews/deep_8.md:2879:For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
./reviews/deep_8.md:2883:𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
./reviews/deep_8.md:2904:Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
./reviews/deep_8.md:2926:where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
./reviews/deep_8.md:2928:Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
./reviews/deep_8.md:2930:Figure 3b. We calculate the EFA loss from the text to its joint embedding, adapted from the triplet
./reviews/deep_8.md:2935:/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modality average precision optimization for visible thermal person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:3020:0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_8.md:5027:前四份文本已经抽出来了。`CoRL` 和 `CAP` 的前 8 页覆盖了方法与部分实验，文本抽取很长有截断，我会再针对方法小节精确抽取，避免只凭摘要判断。
./reviews/deep_8.md:5045:/bin/zsh -lc "pdftotext -f 1 -l 8 'CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:5057:Framework (CVAF) with two training stages. In the first stage, learnable text tokens are employed to represent
./reviews/deep_8.md:5058:identity-aware textual descriptions. To promote consistent alignment across varying viewpoints, we introduce
./reviews/deep_8.md:5059:a Text Consistency Loss (TCL) that regularizes the stability of text-token interactions with multi-view images.
./reviews/deep_8.md:5061:along spatial and channel dimensions. A text-guided cross-attention mechanism generates spatial attention
./reviews/deep_8.md:5062:maps to explicitly emphasize identity-relevant regions, while semantic matching between textual features
./reviews/deep_8.md:5153:image-text pairs spanning diverse scenes, viewpoints, and visual appearances, which implicitly
./reviews/deep_8.md:5160:learnable identity-aware text tokens, optimized via a supervised contrastive loss to align images and
./reviews/deep_8.md:5161:textual descriptions across views. To ensure that these tokens capture consistent identity semantics,
./reviews/deep_8.md:5165:manually labeled attributes. In the second stage, the learned text tokens are reused to guide image
./reviews/deep_8.md:5167:jointly across spatial and channel dimensions. It first applies text-guided cross-attention to generate
./reviews/deep_8.md:5169:tokens based on their semantic similarity to the text features, suppressing background noise and
./reviews/deep_8.md:5176:text tokens, and design a lightweight SFM to emphasize identity-relevant features.
./reviews/deep_8.md:5260:images and texts. Cross-modal alignment has been shown to significantly boost the performance of
./reviews/deep_8.md:5262:loss between image and text features. Thanks to its remarkable cross-modal retrieval accuracy
./reviews/deep_8.md:5265:TCM [35] directly applies CLIP for text detection. CLIP-ReID [10] extends CLIP’s capabilities to
./reviews/deep_8.md:5267:[31] aligns fine-grained image patches with textual descriptions to improve identity discrimination
./reviews/deep_8.md:5268:in text-image retrieval. CSDN [36] leverages CLIP-based semantic cues to bridge modality gaps,
./reviews/deep_8.md:5275:cross-view robustness. In the first stage, a TCL is incorporated to encourage the learnable identityspecific text tokens to capture more generalized and discriminative semantic representations across
./reviews/deep_8.md:5276:diverse viewpoints. In the second stage, a SFM is proposed to parse the textual features and leverage them to guide the image encoder. Through cross-modal attention, SFM enables the model to
./reviews/deep_8.md:5287:ability through a set of learnable text tokens assigned to each identity. These tokens are fed into
./reviews/deep_8.md:5288:the text encoder to form semantically ambiguous yet identity-related textual descriptions, enabling
./reviews/deep_8.md:5289:flexible alignment between image and text representations.
./reviews/deep_8.md:5296:Fig. 2. Comparison between CLIP-ReID and the proposed CVAF. (a) CLIP-ReID employs learnable text tokens
./reviews/deep_8.md:5297:and a two-stage training paradigm for image–text alignment. (b) CVAF extends CLIP-ReID by introducing
./reviews/deep_8.md:5301:training strategy. In the first stage, the image and text encoders inherited from CLIP are frozen,
./reviews/deep_8.md:5302:while the learnable text tokens are trained from scratch using a supervised contrastive loss computed within each batch. This stage aims to establish robust image–text associations by learning
./reviews/deep_8.md:5303:discriminative text tokens for each identity. In the second stage, the learned ID-specific text tokens
./reviews/deep_8.md:5304:and the text encoder are fixed, serving as stable semantic constraints to fine-tune the image encoder.
./reviews/deep_8.md:5314:two novel components: the TCL and SFM. In the first stage, we introduce an identity-level textsupervised view-consistency contrastive learning mechanism. For each identity, 𝑀 learnable text
./reviews/deep_8.md:5315:tokens are initialized and fed into the frozen text encoder to generate ambiguous descriptions.
./reviews/deep_8.md:5316:During this stage, both the image and text encoders from CLIP are kept frozen, and only the
./reviews/deep_8.md:5317:text tokens are optimized from scratch based on the contrastive loss computed within a batch.
./reviews/deep_8.md:5319:text tokens to capture discriminative semantic representations for each identity across different
./reviews/deep_8.md:5320:views. This stage promotes robust cross-modal associations by aligning the text features with the
./reviews/deep_8.md:5323:the deeper layers of the image encoder. During this stage, the learnable text tokens and the text
./reviews/deep_8.md:5325:Through cross-modal attention, the regions of the image corresponding to spatial terms in the text
./reviews/deep_8.md:5334:reweighting mechanism guided by textual semantics to filter out noise interference in the visual
./reviews/deep_8.md:5340:During the first-stage training of learnable text tokens, each identity in the training set is associated
./reviews/deep_8.md:5341:with a textual description, while each description corresponds to multiple images of the same
./reviews/deep_8.md:5342:ID captured from different views. As a result, the learned text tokens are expected to encode
./reviews/deep_8.md:5345:is particularly crucial due to the extreme viewpoint variations. To encourage the text encoder to
./reviews/deep_8.md:5347:is to enforce consistency between the text embedding and image features from different views of
./reviews/deep_8.md:5349:The design of TCL is as follows. For each ID, we collect its associated text feature and a set of
./reviews/deep_8.md:5353:text embedding 𝑡, the similarity between each image feature and the text feature is computed as:
./reviews/deep_8.md:5359:Ideally, we believe that the similarity between different images and text features belonging to
./reviews/deep_8.md:5360:the same ID should be as close as possible, which indicates that the text feature effectively captures
./reviews/deep_8.md:5375:the textual representation to maintain consistent correlations with image features across varying
./reviews/deep_8.md:5381:To enhance the sensitivity of the image encoder to the regions described by text features and
./reviews/deep_8.md:5383:semantic filtering functionalities, as depicted in Figure 3. SFM effectively combines textual features,
./reviews/deep_8.md:5388:After the first-stage training, the CLIP-encoded textual features 𝑡 ∈ R𝐵×𝐷 are obtained. These
./reviews/deep_8.md:5400:cross-modal attention map is computed to measure the semantic relevance between the textual
./reviews/deep_8.md:5402:the textual features contain spatial cues, allowing the model to highlight regions in the image that
./reviews/deep_8.md:5412:where 𝑄 denotes the query vectors derived from the textual features 𝑡 and the learnable projection
./reviews/deep_8.md:5429:Meanwhile, to further refine the image features, a text-guided dynamic channel weighting
./reviews/deep_8.md:5430:mechanism is introduced. The text feature is mapped to a dynamic channel weight vector w𝑐 via a
./reviews/deep_8.md:5438:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./reviews/deep_8.md:5457:/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf' - | sed -n '/3[[:space:]]*Methods/,/4[[:space:]]*Experiments/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:5460:/bin/zsh -lc "pdftotext -f 1 -l 8 'CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:5462:/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modality average precision optimization for visible thermal person re-identification.pdf' - | sed -n '/3\\. Proposed method/,/4\\. Experiments/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:5811:two vectors; 𝒇𝑦𝑖 𝑡 is the text feature with identity 𝑦𝑖 ; 𝑃 (𝑦𝑖 ) denotes a set
./reviews/deep_8.md:5829:they treat the text features corresponding to the visible image as
./reviews/deep_8.md:5830:‘Key’, and the text features corresponding to the infrared image as
./reviews/deep_8.md:5832:improve the semantic richness of text features. However, for clotheschanging person re-identification, there is no modal difference between
./reviews/deep_8.md:5838:biased towards a single text feature, we propose the HSF loss to learn
./reviews/deep_8.md:5845:we balance the contributions of both pseudo-text descriptions using a
./reviews/deep_8.md:5877:token is then combined with the text vectors. Specifically, we design
./reviews/deep_8.md:5878:a mini-network consisting of a two-layer bottleneck structure (LinearReLU-Linear) to project image features into the text space. Let ℎ𝜃 ()
./reviews/deep_8.md:5879:denote the mini-network parameterized by 𝜃, so that each text token
./reviews/deep_8.md:5994:where 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 denote the text features of 𝑇𝑖𝑅 and 𝑇𝑖𝐺 . 𝑇𝑚 (𝑥)
./reviews/deep_8.md:5995:represent the text token of image 𝑥.
./reviews/deep_8.md:6034:text descriptions corresponding to RGB and grayscale images. Upon
./reviews/deep_8.md:6036:to obtain the pseudo-text generated during the first stage. We then
./reviews/deep_8.md:6445:Verify the effectiveness of MDG on PRCC, MI is our Meta-Insert skill.
./reviews/deep_8.md:6630:HSF loss, whose first term guides the visual features to bias the pseudotext space generated by RGB images, and the second term guides the
./reviews/deep_8.md:6631:visual features to bias the pseudo-text space generated by gray-scale
./reviews/deep_8.md:6638:plays the role of fusing descriptions, whose target is enriching textual
./reviews/deep_8.md:6656:richer semantic information, largely due to the inclusion of grayscale images. While RGB images provide pseudo-text focusing on color
./reviews/deep_8.md:6658:and emphasize shape information, such as body shape. When used together in training the MDGs, they generate more instructive pseudo-text
./reviews/deep_8.md:6684:Multi-Description Generator (MDG), which generates pseudo-text descriptions in both RGB and grayscale modes. This dual-mode approach,
./reviews/deep_8.md:6685:combined with a text-image mutual loss mechanism, effectively reduces the color sensitivity of the text encoder. Notably, each linguistic
./reviews/deep_8.md:6742:/bin/zsh -lc "pdftotext -f 8 -l 12 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:6767:𝐿 as the text token 𝑓𝑖𝑡 .
./reviews/deep_8.md:6770:𝐿 as well as image embeddings {𝑓 𝑣 } 𝑁 , retheir corresponding original text embeddings {𝑓𝑖𝑡 }𝑖=1
./reviews/deep_8.md:6773:computation between the joint embeddings and their corresponding text and image embeddings,
./reviews/deep_8.md:6775:present the calculation between the joint embeddings and the text embeddings, while the remaining
./reviews/deep_8.md:6777:For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
./reviews/deep_8.md:6781:𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
./reviews/deep_8.md:6799:Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
./reviews/deep_8.md:6821:where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
./reviews/deep_8.md:6823:Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
./reviews/deep_8.md:6825:in Figure 3(b). We calculate the EFA loss from the text to its joint embedding, adapted from the
./reviews/deep_8.md:6847:Similarly, the EFA loss from the joint embedding to its original text can be computed following
./reviews/deep_8.md:6857:As mentioned, FMFA aims to improve both the global and local cross-modal alignment of image-text
./reviews/deep_8.md:6860:FMFA. The ID loss directly classifies the global features obtained from both the image and the text
./reviews/deep_8.md:6863:cross-modal interaction to obtain a joint embedding, and then predicts the [MASK] text token to
./reviews/deep_8.md:6875:Datasets. We assess FMFA on three widely used text-based person retrieval datasets, following the
./reviews/deep_8.md:6876:data splits introduced in IRRA [21]. CUHK-PEDES [27] contains 40,206 images associated with
./reviews/deep_8.md:6877:13,003 identities, where each image is paired with two textual descriptions. Of these identities,
./reviews/deep_8.md:6881:and reserves 1,000 identities for testing. RSTPReid [60] comprises 20,505 images from 4,101 identities
./reviews/deep_8.md:6890:the identical CLIP-ViT-B/16 model for visual encoding and Xformer for text encoding, following
./reviews/deep_8.md:6891:the setup used in IRRA [21] for our experiments. Specifically, images are resized to 384 × 128 pixels,
./reviews/deep_8.md:6897:The weight factor 𝛼 of A-SDM is set to 10.0 by default, and set to 1 in the RSTPReid dataset, and
./reviews/deep_8.md:6912:RSTPReid
./reviews/deep_8.md:6926:“T to E” means the EFA loss from textual embeddings to
./reviews/deep_8.md:6957:to as IRRA𝑅 , which represents the performance of our reimplementation of the IRRA model. CLIP
./reviews/deep_8.md:6962:Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10. When
./reviews/deep_8.md:6965:NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based backbone
./reviews/deep_8.md:6967:Evaluation Results on RSTPReid. We assess FMFA on the latest RSTPReid benchmark, as presented in Table 4. Using the VL-Backbones without ReID-domain pre-training, FMFA achieves
./reviews/deep_8.md:6969:mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
./reviews/deep_8.md:6971:IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
./reviews/deep_8.md:6994:IRRA𝑅 [21]
./reviews/deep_8.md:7010:APTM [55]
./reviews/deep_8.md:7015:UniPT [41] + IRRA [21]
./reviews/deep_8.md:7017:PLIP [61] + IRRA [21]
./reviews/deep_8.md:7019:NAM [45] + IRRA𝑅
./reviews/deep_8.md:7024:HAM [22] + IRRA
./reviews/deep_8.md:7174:encoder and text encoder, respectively. “IRRA𝑅 ” means the model that we reproduce. Bold denotes best result.
./reviews/deep_8.md:7177:RSTPReid Dataset
./reviews/deep_8.md:7191:IRRA𝑅 [21]
./reviews/deep_8.md:7213:APTM [55]
./reviews/deep_8.md:7217:UniPT [41] + IRRA [21]
./reviews/deep_8.md:7220:PLIP [61] + IRRA [21]
./reviews/deep_8.md:7223:NAM [45] + IRRA𝑅
./reviews/deep_8.md:7230:HAM [22] + IRRA𝑅
./reviews/deep_8.md:7279:IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
./reviews/deep_8.md:7306:IRRA𝑅 [21]
./reviews/deep_8.md:7329:UniPT [41] + IRRA [21]
./reviews/deep_8.md:7332:PLIP [61] + IRRA [21]
./reviews/deep_8.md:7335:NAM [45] + IRRA𝑅
./reviews/deep_8.md:7342:HAM [22] + IRRA𝑅
./reviews/deep_8.md:7464:RSTPReid
./reviews/deep_8.md:7484:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./reviews/deep_8.md:7495:we omit the components of L𝐼 𝐷 and the IRR module that were proposed by IRRA and used in all
./reviews/deep_8.md:7506:/bin/zsh -lc "pdftotext -f 3 -l 5 'Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:7514:Stage 1: Virtual-text Embedding
./reviews/deep_8.md:7556:Virtual-text features
./reviews/deep_8.md:7558:Virtual-text Branch
./reviews/deep_8.md:7586:Visual/Virtualtext Features
./reviews/deep_8.md:7631:Figure 2: Overview of our CoRL. In the first stage, we introduce a Virtual-text Embedding Synthesizer to generate a virtual-text embedding
./reviews/deep_8.md:7632:from each image. The generated virtual-text embedding is then used to replace actual text in modeling multi-modal complementarity with
./reviews/deep_8.md:7634:Consistency Loss, which encourages the exchange of information and mutual enhancement between image and virtual-text features.
./reviews/deep_8.md:7641:models on vast collections of image-text pairs, demonstrating exceptional capabilities in semantic understanding [Shen
./reviews/deep_8.md:7644:achieve a deep understanding of both image and text semantics and their correspondences. Some studies [Li et al.,
./reviews/deep_8.md:7646:texts with tasks such as mask language modeling and image captioning. However, these methods typically necessitate pairwise interactions for all image-text pairs, leading
./reviews/deep_8.md:7650:contrastive representation learning into Vision-language pretraining. These methods encode images and texts separately
./reviews/deep_8.md:7653:image-text pairs. The strong multi-modal alignment capability of CLIP prompts us to introduce it into the TIPR task.
./reviews/deep_8.md:7666:The TIPR dataset contains image-text pairs of persons with
./reviews/deep_8.md:7668:collected from different cameras with their annotated text descriptions. The goal is to accurately identify images that belong to the same identity as a given text description. This requires the TIPR model to extract identity-discriminative representations and accurately establish cross-modal correspondences. To achieve this, we first use a pre-trained CLIP as
./reviews/deep_8.md:7683:Current TIPR methods mainly emphasize image-text associations but overlook their complementarity. Texts offer
./reviews/deep_8.md:7684:the following advantages over images: texts primarily contain pedestrian identity details with less noise. Even when
./reviews/deep_8.md:7685:pedestrians are visually similar, their text descriptions are
./reviews/deep_8.md:7686:often distinct, making text information more discriminative
./reviews/deep_8.md:7687:and robust. Complementing images with textual information
./reviews/deep_8.md:7689:gaps. However, this requires access to corresponding text for
./reviews/deep_8.md:7691:eliminating the dependency on actual text and effectively utilizing text information to enhance image representations. To
./reviews/deep_8.md:7693:develop a Virtual-text Embedding Synthesizer (VES) to generate virtual-text embeddings directly from each image, eliminating the need for actual paired text. In the second stage, we
./reviews/deep_8.md:7695:process to integrate the virtual-text information into images,
./reviews/deep_8.md:7697:Virtual-text Embedding Synthesizer. To eliminate reliance on actual paired text, we propose a Virtual-text Embedding Synthesizer (VES) that generates a virtual-text embedding directly from each image, as illustrated in Figure 2.
./reviews/deep_8.md:7698:CLIP, trained on vast amounts of image-text data, can generate modality-aligned image-text features. We utilize its
./reviews/deep_8.md:7699:alignment capabilities to convert images into text embeddings. VES functions as a CLIP-based encoder-decoder
./reviews/deep_8.md:7705:This embedding is then transformed into virtual-text tokens
./reviews/deep_8.md:7706:through a fully connected layer. Finally, CLIP’s textual encoder Et decodes these virtual-text tokens to generate the
./reviews/deep_8.md:7707:virtual-text embedding ẑt,i .
./reviews/deep_8.md:7708:This generated virtual-text embedding ẑt,i can serve as a
./reviews/deep_8.md:7709:substitute for the actual text embedding to enhance its image
./reviews/deep_8.md:7710:feature, given that the virtual-text and actual-text embeddings
./reviews/deep_8.md:7713:align that aligns the generated virtual-text
./reviews/deep_8.md:7714:embedding with the actual text embedding zt,i at both feature
./reviews/deep_8.md:7716:text Ti .
./reviews/deep_8.md:7761:where zt,i denotes the actual text embedding of text Ti , generated by CLIP’s visual encoder. ∥ · ∥2 denotes the L2 distance,
./reviews/deep_8.md:7763:conversion between images and virtual-text embeddings, we
./reviews/deep_8.md:7829:the virtual-text feature t̂i , we introduce a cross-modal relation consistency loss (CRCL). This loss enforces that the image and virtual-text features share the same relationship with
./reviews/deep_8.md:7832:each batch B, we cluster the image and text features based
./reviews/deep_8.md:7833:on identity labels to obtain visual and textual prototypes, pv
./reviews/deep_8.md:7837:rit between the virtual-text feature t̂i and these textual prototypes pt . Thus, the CRCL is formulated as:
./reviews/deep_8.md:7851:knowledge distillation on instance-to-prototype relations between the visual embedding vi and virtual-text embedding t̂i
./reviews/deep_8.md:7852:allows them to exchange information and benefit from one another. Incorporating virtual-text information emphasizes the
./reviews/deep_8.md:7854:between the image Ii and the text Ti .
./reviews/deep_8.md:7863:VES enables the generation of virtual-text embedding that is
./reviews/deep_8.md:7864:aligned with the actual text for each image, eliminating the
./reviews/deep_8.md:7865:need for actual text.
./reviews/deep_8.md:7866:Cross-modal Collaboration. With the virtual-text embedding, we design a dual-branch cross-modal collaboration
./reviews/deep_8.md:7868:image-text pair (Ii , Ti ), the visual backbone branch generates
./reviews/deep_8.md:7869:the image feature vi of image Ii . Simultaneously, the virtualtext branch first produces a virtual-text embedding ẑt,i from
./reviews/deep_8.md:7871:g(·) to yield a virtual-text feature t̂i adapted to the target domain. Additionally, the text feature ti for text Ti is generated
./reviews/deep_8.md:7872:via the textual backbone. We optimize the Adapter by aligning the virtual-text feature t̂i with the text feature ti through
./reviews/deep_8.md:7883:When a batch contains multiple image-text pairs of the same
./reviews/deep_8.md:7885:all images and texts in the batch. Due to view variations,
./reviews/deep_8.md:7887:can lead to noisy matching of cross-view image-text pairs of
./reviews/deep_8.md:7889:image and its corresponding text form a single-view strong
./reviews/deep_8.md:7890:positive pair, while image-text pairs across different views
./reviews/deep_8.md:7891:are considered as cross-view weak positive pairs. Imagetext pairs with different identities are categorized as negative
./reviews/deep_8.md:7901:with the corresponding text. Thus, each batch contains B
./reviews/deep_8.md:7902:= P K images and their corresponding texts. This results in
./reviews/deep_8.md:7906:similarity for these image-text pairs, denoted as {ssp
./reviews/deep_8.md:7915:guish different image-text pairs, we propose an IBM loss that
./reviews/deep_8.md:7983:we compute the cross entropy loss Lid on image and text features to classify them by identity. This loss forces the network
./reviews/deep_8.md:7992:stage, we train a virtual-text embedding synthesizer, where
./reviews/deep_8.md:7993:CLIP’s visual and textual encoders are frozen and only a fully
./reviews/deep_8.md:7995:V ES to convert image embeddings to virtual-text tokens. We pre-extract and save features from the dataset, allowing us to focus on optimizing this
./reviews/deep_8.md:8002:During inference, both visual and virtual-text features of
./reviews/deep_8.md:8004:to the query text features. The final retrieval score is obtained
./reviews/deep_8.md:8018:2021] consists of 54,522 image-text pairs of 4,102 persons,
./reviews/deep_8.md:8020:pairs from 3,102 people, with the remaining 1,000 people reserved for evaluation. RSTPReid [Zhu et al., 2021] includes
./reviews/deep_8.md:8026:the text sequence is set to 77, and random masking is employed for text augmentation. We use CLIP-ViT-B/16 as the
./reviews/deep_8.md:8031:fine-tune the visual/textual backbones with an initial learning
./reviews/deep_8.md:8048:ICFG-PEDES, and RSTPReid. Our CoRL achieves leading performance on these benchmarks, underscoring its effectiveness and advantages. On the CUHK-PEDES dataset,
./reviews/deep_8.md:8053:the RSTPReid dataset, CoRL delivers impressive results with
./reviews/deep_8.md:8058:cross-modal complementarity and establishing precise crossmodal correspondences. CoRL effectively integrates textual information into images to enhance their discriminability
./reviews/deep_8.md:8060:matching loss to address noisy correspondences among different types of image-text pairs.
./reviews/deep_8.md:8072:/bin/zsh -lc "pdftotext -f 8 -l 12 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:8109:𝐿 as the text token 𝑓𝑖𝑡 .
./reviews/deep_8.md:8116:and their corresponding original text embeddings {𝑓𝑖 }𝑖=1 as well as image embeddings {𝑓𝑖 }𝑖=1 ,
./reviews/deep_8.md:8118:computation between the joint embeddings and their corresponding text and image embeddings,
./reviews/deep_8.md:8120:present the calculation between the joint embeddings and the text embeddings, while the remaining
./reviews/deep_8.md:8122:For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
./reviews/deep_8.md:8126:𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
./reviews/deep_8.md:8147:Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
./reviews/deep_8.md:8169:where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
./reviews/deep_8.md:8171:Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
./reviews/deep_8.md:8173:Figure 3b. We calculate the EFA loss from the text to its joint embedding, adapted from the triplet
./reviews/deep_8.md:8197:Similarly, the EFA loss from the joint embedding to its original text can be computed following
./reviews/deep_8.md:8207:As mentioned, FMFA aims to improve both the global and local cross-modal alignment of image-text
./reviews/deep_8.md:8210:FMFA. The ID loss directly classifies the global features obtained from both the image and the text
./reviews/deep_8.md:8213:implicit cross-modal interaction to obtain a joint embedding, and then predicts the [MASK] text
./reviews/deep_8.md:8225:Datasets. We assess FMFA on three widely used text-based person retrieval datasets, following
./reviews/deep_8.md:8226:the data splits introduced in IRRA [21]. CUHK-PEDES [27] contains 40,206 images associated with
./reviews/deep_8.md:8227:13,003 identities, where each image is paired with two textual descriptions. Of these identities,
./reviews/deep_8.md:8231:and reserves 1,000 identities for testing. RSTPReid [60] comprises 20,505 images from 4,101 identities
./reviews/deep_8.md:8241:employ the identical CLIP-ViT-B/16 model for visual encoding and Xformer for text encoding,
./reviews/deep_8.md:8242:following the setup used in IRRA [21] for our experiments. Specifically, images are resized to
./reviews/deep_8.md:8253:Table 1. The margins utilized in the EFA loss. “T. to E.” means the EFA loss from textual embeddings to
./reviews/deep_8.md:8260:RSTPReid
./reviews/deep_8.md:8288:value of 1.0. The weight factor 𝛼 of A-SDM is set to 10.0 by default, and set to 1 in the RSTPReid
./reviews/deep_8.md:8306:and Table 5 is referred to as IRRA𝑅 , which represents the performance of our reimplementation of
./reviews/deep_8.md:8307:the IRRA model. CLIP means the ViT-B/16 architecture after fine-tuning under the InfoNCE loss
./reviews/deep_8.md:8310:74.16% Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10.
./reviews/deep_8.md:8313:with NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based
./reviews/deep_8.md:8315:Evaluation Results on RSTPReid. We assess FMFA on the latest RSTPReid benchmark, as
./reviews/deep_8.md:8318:mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
./reviews/deep_8.md:8326:“Type” column stand for global-matching method, local-matching method and pre-trained model with ReIDdomain respectively. “Image Enc.” and “Text Enc.” mean the backbone of image encoder and text encoder
./reviews/deep_8.md:8327:respectively. “IRRA𝑅 ” means the model that we reproduce.
./reviews/deep_8.md:8342:IRRA𝑅 [21]
./reviews/deep_8.md:8358:APTM [55]
./reviews/deep_8.md:8363:UniPT [41] + IRRA [21]
./reviews/deep_8.md:8365:PLIP [61] + IRRA [21]
./reviews/deep_8.md:8367:NAM [45] + IRRA𝑅
./reviews/deep_8.md:8371:HAM [22] + IRRA𝑅
./reviews/deep_8.md:8520:Table 4. Comparisons with state-of-the-art methods on the RSTPReid dataset.
./reviews/deep_8.md:8534:IRRA𝑅 [21]
./reviews/deep_8.md:8556:APTM [55]
./reviews/deep_8.md:8560:UniPT [41] + IRRA [21]
./reviews/deep_8.md:8563:PLIP [61] + IRRA [21]
./reviews/deep_8.md:8566:NAM [45] + IRRA𝑅
./reviews/deep_8.md:8573:HAM [22] + IRRA𝑅
./reviews/deep_8.md:8618:IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
./reviews/deep_8.md:8649:IRRA𝑅 [21]
./reviews/deep_8.md:8672:UniPT [41] + IRRA [21]
./reviews/deep_8.md:8675:PLIP [61] + IRRA [21]
./reviews/deep_8.md:8678:NAM [45] + IRRA𝑅
./reviews/deep_8.md:8685:HAM [22] + IRRA𝑅
./reviews/deep_8.md:8729:IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
./reviews/deep_8.md:8731:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./reviews/deep_8.md:8742:we omit the components of L𝑖𝑑 and the IRR module that were proposed by IRRA and used in all
./reviews/deep_8.md:8756:0.58% decline in mAP on the RSTPReid dataset. These results collectively validate the consistent
./reviews/deep_8.md:8762:/bin/zsh -lc "pdftotext -f 8 -l 11 'CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:8770:cross-modal attention map is computed to measure the semantic relevance between the textual
./reviews/deep_8.md:8772:the textual features contain spatial cues, allowing the model to highlight regions in the image that
./reviews/deep_8.md:8782:where 𝑄 denotes the query vectors derived from the textual features 𝑡 and the learnable projection
./reviews/deep_8.md:8799:Meanwhile, to further refine the image features, a text-guided dynamic channel weighting
./reviews/deep_8.md:8800:mechanism is introduced. The text feature is mapped to a dynamic channel weight vector w𝑐 via a
./reviews/deep_8.md:8808:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./reviews/deep_8.md:8837:losses: the image-to-text loss L𝑖2𝑡 and the text-to-image loss L𝑡 2𝑖 , which enforce bidirectional
./reviews/deep_8.md:8838:image-text alignment. Both losses share the same supervised contrastive form:
./reviews/deep_8.md:8851:and 𝜏 is the temperature. Specifically, L𝑖2𝑡 = L (image, text) and L𝑡 2𝑖 = L (text, image). These two
./reviews/deep_8.md:8854:image-text pairs achieve high similarity. Without these supervised contrastive objectives, TCL
./reviews/deep_8.md:8856:the intra-class variance is large. Furthermore, we incorporate the text-to-image cross-entropy
./reviews/deep_8.md:8888:Tesla A40 GPU. In the first training stage, we adopt the text encoder from CLIP as the backbone
./reviews/deep_8.md:8889:for extracting textual features, and the number of learnable text tokens 𝑀 is set to 4. The model is
./reviews/deep_8.md:9107:view-invariant textual representations for the same identity, thereby providing stable semantic
./reviews/deep_8.md:9118:semantic alignment between image and text features. Moreover, Adding the SFM alone brings more
./reviews/deep_8.md:9267:indicates that the learnable text tokens are capable of aggregating multi-view information from
./reviews/deep_8.md:9269:view-invariant features as textual descriptions. These view-invariant text embeddings then serve as
./reviews/deep_8.md:9275:/bin/zsh -lc "pdftotext -f 3 -l 5 'CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:9281:Fig. 2. Overview of the proposed method. Our CSGN includes two stages. Stage 1: Training MDG using RGB maps and gray-scale maps with Meta-Insert skill. Stage 2: Training
./reviews/deep_8.md:9292:text descriptions for each image. To address this challenge, we introduce a novel framework named CSGN, which consists of three core
./reviews/deep_8.md:9296:between text and image representations, effectively mitigating the
./reviews/deep_8.md:9297:text encoder’s sensitivity to color variations. In the second stage, VSS
./reviews/deep_8.md:9303:to generating discriminative textual and visual representations for ReID
./reviews/deep_8.md:9314:process ensures a one-to-one correspondence between images and texts.
./reviews/deep_8.md:9320:concrete text labels, T(⋅) first converts each word into a unique numeric
./reviews/deep_8.md:9323:To enable parallel computation, each text sequence has a fixed length
./reviews/deep_8.md:9326:token is considered as the feature representation of the text, which
./reviews/deep_8.md:9333:achieve the best alignment between the visual and text features.
./reviews/deep_8.md:9341:the dataloader contains 𝑁 (text-image) pairs, we first encode these 𝑁
./reviews/deep_8.md:9342:texts into one-dimensional vectors of length d using the Text Encoder.
./reviews/deep_8.md:9343:This batch of text data is represented as {𝑇1 , 𝑇2 , … , 𝑇𝑁 }. Similarly,
./reviews/deep_8.md:9345:where each image vector has the same length as the text vector. The
./reviews/deep_8.md:9451:two vectors; 𝒇𝑦𝑖 𝑡 is the text feature with identity 𝑦𝑖 ; 𝑃 (𝑦𝑖 ) denotes a set
./reviews/deep_8.md:9469:they treat the text features corresponding to the visible image as
./reviews/deep_8.md:9470:‘Key’, and the text features corresponding to the infrared image as
./reviews/deep_8.md:9472:improve the semantic richness of text features. However, for clotheschanging person re-identification, there is no modal difference between
./reviews/deep_8.md:9478:biased towards a single text feature, we propose the HSF loss to learn
./reviews/deep_8.md:9485:we balance the contributions of both pseudo-text descriptions using a
./reviews/deep_8.md:9517:token is then combined with the text vectors. Specifically, we design
./reviews/deep_8.md:9518:a mini-network consisting of a two-layer bottleneck structure (LinearReLU-Linear) to project image features into the text space. Let ℎ𝜃 ()
./reviews/deep_8.md:9519:denote the mini-network parameterized by 𝜃, so that each text token
./reviews/deep_8.md:9634:where 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 denote the text features of 𝑇𝑖𝑅 and 𝑇𝑖𝐺 . 𝑇𝑚 (𝑥)
./reviews/deep_8.md:9635:represent the text token of image 𝑥.
./reviews/deep_8.md:9674:text descriptions corresponding to RGB and grayscale images. Upon
./reviews/deep_8.md:9676:to obtain the pseudo-text generated during the first stage. We then
./reviews/deep_8.md:9901:/bin/zsh -lc "pdftotext -f 12 -l 15 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:9927:IRRA𝑅 [21]
./reviews/deep_8.md:9950:UniPT [41] + IRRA [21]
./reviews/deep_8.md:9953:PLIP [61] + IRRA [21]
./reviews/deep_8.md:9956:NAM [45] + IRRA𝑅
./reviews/deep_8.md:9963:HAM [22] + IRRA𝑅
./reviews/deep_8.md:10085:RSTPReid
./reviews/deep_8.md:10105:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./reviews/deep_8.md:10116:we omit the components of L𝐼 𝐷 and the IRR module that were proposed by IRRA and used in all
./reviews/deep_8.md:10135:respectively, as well as a 1.55% drop in Rank-5 and a 0.58% decline in mAP on the RSTPReid dataset.
./reviews/deep_8.md:10143:RSTPReid dataset. However, EFA causes a 0.11% drop in Rank-5 on ICFG-PEDES, suggesting that
./reviews/deep_8.md:10150:and 0.95% drops in Rank-1 on CUHK-PEDES, ICFG-PEDES, and RSTPReid, and causes a 1.23% drop
./reviews/deep_8.md:10292:EFA explicitly aggregates image patches and text tokens, allowing us to observe whether the most
./reviews/deep_8.md:10301:on RSTPReid to 50 s vs. 91 s on ICFG-PEDES. These comparisons clearly demonstrate that FMFA
./reviews/deep_8.md:10304:baseline IRRA𝑅 and our proposed FMFA on the CUHK-PEDES dataset. Figure 6 illustrates that FMFA
./reviews/deep_8.md:10306:For query texts where the baseline performs well, FMFA further improves performance by retrieving
./reviews/deep_8.md:10316:/bin/zsh -lc "pdftotext -f 11 -l 14 'CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:10464:indicates that the learnable text tokens are capable of aggregating multi-view information from
./reviews/deep_8.md:10466:view-invariant features as textual descriptions. These view-invariant text embeddings then serve as
./reviews/deep_8.md:10543:validating that enhancing the view-invariance of text token representations is an effective strategy
./reviews/deep_8.md:10551:features and the learned text embeddings. In our framework, each textual description corresponds
./reviews/deep_8.md:10553:text embedding should align equally well with all associated image features, regardless of view.
./reviews/deep_8.md:10554:Therefore, we hypothesize that applying TCL should reduce the variance of image-text similarities
./reviews/deep_8.md:10557:their fixed textual embeddings and image features under different views, both with and without TCL.
./reviews/deep_8.md:10558:For each identity, we report the mean similarity and variance across multi-view image–text pairs,
./reviews/deep_8.md:10567:this to the model becoming more confident in aligning the text embedding with certain dominant
./reviews/deep_8.md:10573:textual representation.
./reviews/deep_8.md:10580:Fig. 4. Comparison of image-text similarity before and after applying TCL. (a) Mean similarity for selected
./reviews/deep_8.md:10591:that, guided by the learned text representations, SFM effectively enhances the image encoder’s
./reviews/deep_8.md:10609:is because shallow features primarily capture low-level textures and spatial details, which are
./reviews/deep_8.md:10714:/bin/zsh -lc "pdftotext -f 7 -l 9 'CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:10721:Verify the effectiveness of MDG on PRCC, MI is our Meta-Insert skill.
./reviews/deep_8.md:10906:HSF loss, whose first term guides the visual features to bias the pseudotext space generated by RGB images, and the second term guides the
./reviews/deep_8.md:10907:visual features to bias the pseudo-text space generated by gray-scale
./reviews/deep_8.md:10914:plays the role of fusing descriptions, whose target is enriching textual
./reviews/deep_8.md:10932:richer semantic information, largely due to the inclusion of grayscale images. While RGB images provide pseudo-text focusing on color
./reviews/deep_8.md:10934:and emphasize shape information, such as body shape. When used together in training the MDGs, they generate more instructive pseudo-text
./reviews/deep_8.md:10960:Multi-Description Generator (MDG), which generates pseudo-text descriptions in both RGB and grayscale modes. This dual-mode approach,
./reviews/deep_8.md:10961:combined with a text-image mutual loss mechanism, effectively reduces the color sensitivity of the text encoder. Notably, each linguistic
./reviews/deep_8.md:11017:Sung, Yun-Hsuan, Li, Zhen, Duerig, Tom, 2021. Scaling up visual and visionlanguage representation learning with noisy text supervision. In: International
./reviews/deep_8.md:11043:for image re-identification without concrete text labels. In: Proceedings of the
./reviews/deep_8.md:11133:/bin/zsh -lc "pdftotext -f 8 -l 10 'Cross-modality average precision optimization for visible thermal person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_8.md:12105:1. 触发观察：作者抓住两个具体问题。第一，TIPR 过去总在做图文“关联”，但文本本身比图像更干净，身份细节更集中，能补足视觉噪声。第二，同一身份的图文正样本并不等价，单视角对应文本是强正样本，跨视角同身份文本可能有外观差异，是弱正样本。
./reviews/deep_8.md:12106:2. 重定义动作：他们把问题从“图文对齐”改写成“跨模态互补协作”，再把正样本从一个类别拆成“强正、弱正、负样本”的有界匹配问题。关键词是 `complementarity`、`virtual-text`、`collaboration`、`identity-bounded matching`。
./reviews/deep_8.md:12107:3. 机制怎么长出来：如果文本有用但推理时图像没有配套文本，就自然推出 `VES`，从图像合成虚拟文本特征；如果虚拟文本要真有用，就要和真实文本特征对齐，再用 `CRCL` 让视觉特征和虚拟文本特征对原型的关系一致。强弱正样本的观察则直接推出 `IBM`，让强正相似度高于上界，弱正落在中间，负样本低于下界。
./reviews/deep_8.md:12108:4. 证据闭环：关键不是总表涨点，而是 `IBM*` 去掉强弱正区分后掉点，证明“正样本内部有层级”这个重定义有意义；`VES` 的各个对齐损失消融证明虚拟文本不是随便加的噪声；检索可视化展示 CoRL 能把视觉相似但文本语义不同的人区分开。
./reviews/deep_8.md:12133:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
./reviews/deep_8.md:12134:2. 重定义动作：作者把 CCReID 从“学习衣服不变视觉特征”改写成“用异质语义描述引导视觉特征摆脱衣服偏置”。关键词是 `semantic guidance`、`visual bottleneck`、`multi-description`、`RGB and gray-scale pseudo-text`、`heterogeneous semantic fusion`。
./reviews/deep_8.md:12135:3. 机制怎么长出来：既然 RGB 文本偏颜色，灰度图可以弱化颜色并突出体型、姿态，就自然推出 `MDG` 同时生成 RGB 和灰度伪文本；既然两种文本各有偏向，就用 `HSF` 同时约束视觉特征；`VSS` 的 ResNet+ViT 融合是补强视觉编码，和核心重定义的绑定不如 MDG/HSF 紧。
./reviews/deep_8.md:12142:3. 机制怎么长出来：如果同一身份的文本 token 应该稳定对齐多个视角图像，就自然推出 `TCL`，约束同一身份多视角图像与文本的相似度方差；如果文本 token 已经学到稳定身份语义，就用 `SFM` 通过文本引导的空间注意力和通道重权重过滤图像 patch。
./reviews/deep_8.md:12143:4. 证据闭环：表 3、表 4 在 AG-ReID 和 CARGO 上分别消融 `TCL`、`SFM` 和组合，且 ViT/ResNet 都有效；更关键的是他们额外画了图文相似度均值和方差，证明 `TCL` 确实降低同一身份跨视角相似度波动；`SFM` 还做了空间、通道、残差和插入层位置消融。
./reviews/deep_8.md:12152:1. 触发观察：作者抓住两个具体问题。第一，TIPR 过去总在做图文“关联”，但文本本身比图像更干净，身份细节更集中，能补足视觉噪声。第二，同一身份的图文正样本并不等价，单视角对应文本是强正样本，跨视角同身份文本可能有外观差异，是弱正样本。
./reviews/deep_8.md:12153:2. 重定义动作：他们把问题从“图文对齐”改写成“跨模态互补协作”，再把正样本从一个类别拆成“强正、弱正、负样本”的有界匹配问题。关键词是 `complementarity`、`virtual-text`、`collaboration`、`identity-bounded matching`。
./reviews/deep_8.md:12154:3. 机制怎么长出来：如果文本有用但推理时图像没有配套文本，就自然推出 `VES`，从图像合成虚拟文本特征；如果虚拟文本要真有用，就要和真实文本特征对齐，再用 `CRCL` 让视觉特征和虚拟文本特征对原型的关系一致。强弱正样本的观察则直接推出 `IBM`，让强正相似度高于上界，弱正落在中间，负样本低于下界。
./reviews/deep_8.md:12155:4. 证据闭环：关键不是总表涨点，而是 `IBM*` 去掉强弱正区分后掉点，证明“正样本内部有层级”这个重定义有意义；`VES` 的各个对齐损失消融证明虚拟文本不是随便加的噪声；检索可视化展示 CoRL 能把视觉相似但文本语义不同的人区分开。
./reviews/deep_8.md:12180:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
./reviews/deep_8.md:12181:2. 重定义动作：作者把 CCReID 从“学习衣服不变视觉特征”改写成“用异质语义描述引导视觉特征摆脱衣服偏置”。关键词是 `semantic guidance`、`visual bottleneck`、`multi-description`、`RGB and gray-scale pseudo-text`、`heterogeneous semantic fusion`。
./reviews/deep_8.md:12182:3. 机制怎么长出来：既然 RGB 文本偏颜色，灰度图可以弱化颜色并突出体型、姿态，就自然推出 `MDG` 同时生成 RGB 和灰度伪文本；既然两种文本各有偏向，就用 `HSF` 同时约束视觉特征；`VSS` 的 ResNet+ViT 融合是补强视觉编码，和核心重定义的绑定不如 MDG/HSF 紧。
./reviews/deep_8.md:12189:3. 机制怎么长出来：如果同一身份的文本 token 应该稳定对齐多个视角图像，就自然推出 `TCL`，约束同一身份多视角图像与文本的相似度方差；如果文本 token 已经学到稳定身份语义，就用 `SFM` 通过文本引导的空间注意力和通道重权重过滤图像 patch。
./reviews/deep_8.md:12190:4. 证据闭环：表 3、表 4 在 AG-ReID 和 CARGO 上分别消融 `TCL`、`SFM` 和组合，且 ViT/ResNet 都有效；更关键的是他们额外画了图文相似度均值和方差，证明 `TCL` 确实降低同一身份跨视角相似度波动；`SFM` 还做了空间、通道、残差和插入层位置消融。
./reviews/deep_13.md:14:你是 ReID 创新方法论拆解员 13。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_13.md:36:我会先按项目规则确认当前目标和状态文件，然后只读这批 PDF 的前 8 页文本来拆方法论。这个任务不需要写文件，也不需要 Python。
./reviews/deep_13.md:102:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_13.md:108:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_13.md:137:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_13.md:152:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_13.md:162:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_13.md:171:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_13.md:177:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_13.md:192:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_13.md:202:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_13.md:204:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_13.md:226:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_13.md:227:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_13.md:237:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_13.md:244:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_13.md:257:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_13.md:261:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_13.md:305:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_13.md:312:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_13.md:355:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_13.md:368:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_13.md:460:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_13.md:473:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_13.md:500:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_13.md:507:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_13.md:662:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_13.md:664:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_13.md:685:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_13.md:701:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_13.md:759:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_13.md:760:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_13.md:786:我已经对齐了 `GOAL.md` 和当前状态：现在确实处在读文献、拆别人如何构造创新的阶段。下面先提取每篇前 8 页的章节结构，确认方法部分在文本里的位置，再逐篇读方法。
./reviews/deep_13.md:800:  pdftotext -f 1 -l 8 "$f" - | rg -n "''^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|VI\\.|VII\\.|1\\.|2\\.|3\\.|4\\.|5\\.|6\\.|[0-9]+\\s+[A-Z]|[A-Z][A-Za-z -]+"'$|A'"\\. |B\\. |C\\. |D\\. |E\\. |F\\. )\" | head -n 80
./reviews/deep_13.md:1140:我先从第一篇开始按“引言里的失败叙事，加方法里的机制，实验里的归因证据”三块读。方法本身会用前 8 页文本，消融如果落在后面页面，我会额外读对应页来补证据闭环。
./reviews/deep_13.md:1150:/bin/zsh -lc "pdftotext -f 1 -l 8 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '35,172p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:1294:/bin/zsh -lc "pdftotext -f 1 -l 8 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '290,608p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:1648:/bin/zsh -lc "pdftotext -f 8 -l 10 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:1924:/bin/zsh -lc "pdftotext -f 1 -l 8 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' - | nl -ba | sed -n '49,181p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:1960:    83	0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_13.md:2087:/bin/zsh -lc "pdftotext -f 1 -l 8 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' - | nl -ba | sed -n '273,662p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:2487:/bin/zsh -lc "pdftotext -f 8 -l 13 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' - | nl -ba | rg -n \"4\\.4|4\\.5|4\\.6|Ablation|ablation|Comparison with data augmentation|Generalization|CES|CCP|HFIA|Table|Fig|Visualization|analysis\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:2866:/bin/zsh -lc "pdftotext -f 1 -l 8 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '43,145p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:2902:    77	0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_13.md:2973:/bin/zsh -lc "pdftotext -f 1 -l 8 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '243,580p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:3321:/bin/zsh -lc "pdftotext -f 7 -l 12 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | rg -n \"4\\.4|Ablation|ablation|DTA|DFA|DAW|DWA|LPCD|DGA|GAE|Comparison|Table|Fig|visual|parameter|Runtime|Effect|Baseline|teacher|student|gamma|scale\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:3693:/bin/zsh -lc "pdftotext -f 8 -l 10 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '300,470p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:3872:/bin/zsh -lc "pdftotext -f 9 -l 10 'GAE-Net - A gait-assisted enhancement network for video-based person-reidentification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:3881:/bin/zsh -lc "pdftotext -f 9 -l 10 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:4181:/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '30,148p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:4183:/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '278,683p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:4242:    87	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
./reviews/deep_13.md:4313:   285	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
./reviews/deep_13.md:4649:   621	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
./reviews/deep_13.md:4718:/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '672,815p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:4870:/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '803,850p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:4888:   819	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
./reviews/deep_13.md:4926:/bin/zsh -lc "pdftotext -f 8 -l 12 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|GHOP|AHOP|AFA|high-order|order|Table|Market|MSMT|Duke|components|Visualization|epsilon|group|shuffle\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:5149:474-   474	“GS,” AND “EA” R EPRESENT H IGH -O RDER P OOLING , M ULTIPLE
./reviews/deep_13.md:5208:/bin/zsh -lc "pdftotext -f 11 -l 12 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '1,220p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:5274:    65	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
./reviews/deep_13.md:5293:    84	“GS,” AND “EA” R EPRESENT H IGH -O RDER P OOLING , M ULTIPLE
./reviews/deep_13.md:5404:/bin/zsh -lc "pdftotext -f 1 -l 8 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' - | nl -ba | sed -n '42,154p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:5442:    78	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_13.md:5521:/bin/zsh -lc "pdftotext -f 1 -l 8 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' - | nl -ba | sed -n '241,668p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:5959:/bin/zsh -lc "pdftotext -f 6 -l 12 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' - | nl -ba | rg -n \"Ablation|ablation|CGS|GRSR|GSANet|matching triplet|triplet|generalization|Visualization|Table|Fig|Sampler|PK|Graph|sparse|feature reconstruction|Market|MSMT|improvement|impact\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:6564:/bin/zsh -lc "pdftotext -f 1 -l 8 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | sed -n '48,167p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:6604:    86	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_13.md:6692:/bin/zsh -lc "pdftotext -f 1 -l 8 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | sed -n '220,570p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:7083:/bin/zsh -lc "pdftotext -f 8 -l 12 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|gradient consistency|consistency|global aggregated|GAGA|Table|Fig|iterations|early|late|transfer|black-box|white-box|Visualization|parameter|inner|m |beta|Input transformation|ensemble\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:7335:1183:  1183	attack detection in person re-identiﬁcation using context inconsistency, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021,
./reviews/deep_13.md:7418:/bin/zsh -lc "pdftotext -f 10 -l 10 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | sed -n '1,180p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:7465:/bin/zsh -lc "pdftotext -f 9 -l 9 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | tail -n 220" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:14:你是 ReID 创新方法论拆解员 18。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_18.md:36:我会先按项目规则确认当前目标和状态，再用 `pdftotext` 逐篇读方法部分。因为这是 PDF 阅读任务，我也会先看一下本地 PDF 技能的约束，避免漏掉必要的提取和核对步骤。
./reviews/deep_18.md:58:/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:65:# PDF Skill
./reviews/deep_18.md:79:3. Use `pdfplumber` or `pypdf` for text extraction and quick checks; do not rely on text extraction for layout fidelity.
./reviews/deep_18.md:134:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reviews/deep_18.md:179:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_18.md:194:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_18.md:204:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_18.md:214:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_18.md:220:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_18.md:235:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_18.md:245:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_18.md:247:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_18.md:259:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_18.md:275:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_18.md:431:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_18.md:452:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_18.md:468:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_18.md:526:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_18.md:527:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_18.md:623:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_18.md:636:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_18.md:663:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_18.md:670:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_18.md:743:第一，做训练前两轮独立审查里的 Codex 这一轮。一个改了方法的实验，在开始训练之前，主 agent 会用 `codex exec` 起你来做一次独立代码审查。你看不到另一个审查者的结论，也不知道这是第几轮，也不要假设自己知道改了什么。审查范围是设计文档、新增和改动代码、配置、对照组变量隔离、评测协议和数据流。
./reviews/deep_18.md:745:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reviews/deep_18.md:761:凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_18.md:779:自己写的评测脚本或分析脚本，要先用它复现一个已知的基线成绩。对得上之后，才能用它的结果下结论。新的评测口径、新的度量、新的脚本都按这条执行。
./reviews/deep_18.md:848:任何改了模型或者有新设计的实验，在启动训练之前，都要经过两轮互不通气的独立审查。一轮由 Claude 做，一轮由 Codex 做。两个审查者互相看不到对方结论，也不知道这是第几轮。
./reviews/deep_18.md:850:Codex 这一轮要完整审一遍，不是只看某几处。要看设计文档、新增和改动的代码、配置、对照组、变量隔离和评测协议。结论写进 `experiments/exp{编号}/review-codex.md`。
./reviews/deep_18.md:893:重大决策前，尽量让两个独立视角辩一辩。一个为方案 A 辩护，一个为方案 B 辩护，从技术可行性、创新性、论文价值、风险、成本几方面讲，并给出信心分。最后综合判断，把结论写进决策记录。
./reviews/deep_18.md:923:绝对不要用宽泛匹配的 `pkill -f codex`，那会误杀 Mac 上的 Codex.app。要杀进程就精确匹配。
./reviews/deep_18.md:937:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reviews/deep_18.md:975:/bin/zsh -lc "pdftotext -f 1 -l 8 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|adapter|domain|mixture|parameter'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:977:/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|mix|modality|paradigm'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:1004:66:    66	In the context of DG ReID, the training and testing of models are performed on diverse domains,
./reviews/deep_18.md:1022:87:    87	Adapter [10] in our method. CLIP is a powerful pretrained model which aligns texts and images
./reviews/deep_18.md:1039:127:   127	address domain shifts by aligning text and image features in a shared space, thereby enhancing
./reviews/deep_18.md:1040:130:   130	methods, specifically Adapters, within the context of DG ReID. Adapters serve as domainspecific experts in our framework, demonstrating their efficacy in reducing the number of
./reviews/deep_18.md:1114:340:   340	reason might be that the image-to-text loss L𝑖2𝑡 and ID loss L𝑖𝑑 proposed by CLIP are both inherently cross-entropy losses, thus can provide approximate constraints and optimization capabilities.
./reviews/deep_18.md:1128:376:   376	text features from different domains together to perform L𝑖2𝑡𝑐𝑒 . And the loss for global adapter is
./reviews/deep_18.md:1274:/bin/zsh -lc "pdftotext -f 1 -l 8 'MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|hairstyle|cloth|MSP|reid'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:1319:98:    98	clothing pixels, suppressing texture bias while maintaining body
./reviews/deep_18.md:1354:422:   422	spatial context. To address this limitation, we propose Cloth-Preserved
./reviews/deep_18.md:1427:1026:  1026	pixels within clothing regions while retaining geometric and contextual
./reviews/deep_18.md:1459:/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|mix|modality|paradigm'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:1527:188:   188	context, the Visible-Infrared person re-identification (VI-ReID) task, which was first proposed by
./reviews/deep_18.md:1603:/bin/zsh -lc "pdftotext -f 1 -l 8 'Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|meta|pairwise|relationship|distillation'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:1627:166:   166	a graph structure is more suitable to estimate pairwise labels, as shown in Figure 1 (b), which exploits contextual
./reviews/deep_18.md:1741:/bin/zsh -lc "pdftotext -f 1 -l 8 'Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|memory|meta|shuffled'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:1830:306:   306	context for a dedicated temporal memory module, which aggregates
./reviews/deep_18.md:1940:定位结果显示，6 个文件都能用 `pdftotext` 正常提取。两个 Mix-Modality 文件是一篇论文的 arXiv 版和 ACM 版，正文高度一致，但我会按两个文件分别核对，因为用户明确列了两个文件。
./reviews/deep_18.md:1954:/bin/zsh -lc "pdftotext -f 1 -l 8 'MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '364,675p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:2014:   422	spatial context. To address this limitation, we propose Cloth-Preserved
./reviews/deep_18.md:2270:/bin/zsh -lc "pdftotext -f 1 -l 8 'Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '332,886p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:2272:/bin/zsh -lc "pdftotext -f 1 -l 8 'Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '305,715p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:3258:/bin/zsh -lc "pdftotext -f 1 -l 8 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '211,980p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:3265:   216	both text encoder T and image encoder I of CLIP. Similar to CLIP-ReID, we assign ID-specific
./reviews/deep_18.md:3266:   217	tokens to each person. Specifically, the text descriptions fed into T are designed as “A photo of a
./reviews/deep_18.md:3267:   218	[𝑋 ] 1 [𝑋 ] 2 [𝑋 ] 3 ...[𝑋 ] 𝑀 person,” where each [𝑋 ]𝑚 is a learnable text token with the same dimension
./reviews/deep_18.md:3268:   219	as word embedding. By optimization of two contrastive losses inspired by CLIP (image-to-text loss
./reviews/deep_18.md:3269:   220	L𝑖2𝑡 and modified text-to-image loss L𝑡 2𝑖 ), the trained ID-specific tokens can provide discriminative
./reviews/deep_18.md:3298:   249	where 𝑉𝑖 and𝑇𝑖 are [CLS] tokens of image feature and text feature produced by I and T , respectively.
./reviews/deep_18.md:3301:   252	positive image samples. Therefore, text-to-image loss is modified to:
./reviews/deep_18.md:3371:   322	Similar to CLIP-ReID, we use the ID-specific text features obtained in the first training stage to
./reviews/deep_18.md:3372:   323	calculate the image-to-text cross-entropy L𝑖2𝑡𝑐𝑒 with label smoothing:
./reviews/deep_18.md:3389:   340	reason might be that the image-to-text loss L𝑖2𝑡 and ID loss L𝑖𝑑 proposed by CLIP are both inherently cross-entropy losses, thus can provide approximate constraints and optimization capabilities.
./reviews/deep_18.md:3425:   376	text features from different domains together to perform L𝑖2𝑡𝑐𝑒 . And the loss for global adapter is
./reviews/deep_18.md:3475:/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | sed -n '203,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:3736:/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf' - | nl -ba | sed -n '223,780p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:4101:/bin/zsh -lc "pdftotext -f 1 -l 12 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | sed -n '453,900p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:4103:/bin/zsh -lc "pdftotext -f 1 -l 12 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '388,880p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:4457:   501	First. Image-to-text cross-entropy loss L𝑖2𝑡𝑐𝑒 . This part is the same as the optimization of expert
./reviews/deep_18.md:4625:   669	We adopt the image encoder and text encoder of pretrained CLIP model as our backbone. For the
./reviews/deep_18.md:4627:   671	768 dimensions. The parameters of both the image encoder and the text encoder are frozen. The
./reviews/deep_18.md:4797:/bin/zsh -lc "pdftotext -f 1 -l 12 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf' - | nl -ba | sed -n '556,980p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:5240:/bin/zsh -lc "pdftotext -f 1 -l 10 'MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '675,1020p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:5561:   994	rankings, rather than overfitting to textures or colors.
./reviews/deep_18.md:5582:  1015	confusion from similar colors or textures. The feature maps exhibit a
./reviews/deep_18.md:5590:/bin/zsh -lc "pdftotext -f 1 -l 10 'Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '715,1115p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:5995:/bin/zsh -lc "pdftotext -f 1 -l 12 'Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '886,1380p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:6490:  1379	contexts. Moreover, when combined with SML’s varied query-support
./reviews/deep_18.md:6504:/bin/zsh -lc "pdftotext -f 1 -l 18 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | rg -n -i 'mixing ratio|modality confusion|ablation|hyperparameter|visualization|t-SNE|MBSOS|CIDHL|\\+M|\\+C|Rank-1|mINP|performance degradation'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:6642:/bin/zsh -lc "pdftotext -f 1 -l 18 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | rg -n -i 'ablation|parameter|adapter|global adapter|voting|ID loss|frozen|tunable|visual|baseline|Table 8|Protocol|performance|efficiency'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:6649:16:    16	pre-trained model CLIP, which exploits both visual and text encoders, to learn more robust representations
./reviews/deep_18.md:6665:87:    87	Adapter [10] in our method. CLIP is a powerful pretrained model which aligns texts and images
./reviews/deep_18.md:6683:130:   130	methods, specifically Adapters, within the context of DG ReID. Adapters serve as domainspecific experts in our framework, demonstrating their efficacy in reducing the number of
./reviews/deep_18.md:6696:185:   185	tasks. And we will explore the performance of CLIP in the DG ReID context based on CLIP-ReID.
./reviews/deep_18.md:6730:340:   340	reason might be that the image-to-text loss L𝑖2𝑡 and ID loss L𝑖𝑑 proposed by CLIP are both inherently cross-entropy losses, thus can provide approximate constraints and optimization capabilities.
./reviews/deep_18.md:6744:376:   376	text features from different domains together to perform L𝑖2𝑡𝑐𝑒 . And the loss for global adapter is
./reviews/deep_18.md:6779:671:   671	768 dimensions. The parameters of both the image encoder and the text encoder are frozen. The
./reviews/deep_18.md:6877:1386:  1386	block-aware voting network designed to fully leverage both the image and text encoders of CLIP
./reviews/deep_18.md:6889:1435:  1435	Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In
./reviews/deep_18.md:6912:/bin/zsh -lc "pdftotext -f 1 -l 18 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | sed -n '1563,1720p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:7074:/bin/zsh -lc "pdftotext -f 1 -l 18 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '1108,1280p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:7251:/bin/zsh -lc "pdftotext -f 1 -l 18 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '1350,1385p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:7258:  1355	ID-specific tokens from different datasets are derived from the same shared text feature space, they
./reviews/deep_18.md:7291:已按本地 PDF 用 `pdftotext` 读了方法和关键实验分析。你列的是 6 个文件，其中 Mix-Modality 有 arXiv 版和 ACM 正式版，方法基本一致，但我按两篇分别拆。
./reviews/deep_18.md:7329:3. **机制怎么长出来**：机制很顺。参数效率要求冻结 CLIP，只训 adapter。DG 要保留源域互补性，所以每个源域一个 domain-specific adapter，再加一个 global adapter。ViT 的 adapter 在每个 block 里，如果按传统专家投票要多次 forward，于是提出 block-aware voting，在每个 block 用 CLIP 学到的 ID-specific text tokens 作为 domain prototypes 来算权重。
./reviews/deep_18.md:7346:已按本地 PDF 用 `pdftotext` 读了方法和关键实验分析。你列的是 6 个文件，其中 Mix-Modality 有 arXiv 版和 ACM 正式版，方法基本一致，但我按两篇分别拆。
./reviews/deep_18.md:7384:3. **机制怎么长出来**：机制很顺。参数效率要求冻结 CLIP，只训 adapter。DG 要保留源域互补性，所以每个源域一个 domain-specific adapter，再加一个 global adapter。ViT 的 adapter 在每个 block 里，如果按传统专家投票要多次 forward，于是提出 block-aware voting，在每个 block 用 CLIP 学到的 ID-specific text tokens 作为 domain prototypes 来算权重。
./reviews/deep_25.md:14:你是 ReID 创新方法论拆解员 25。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_25.md:36:我会先按项目规则读目标和状态文件，再逐篇用 `pdftotext -f 1 -l 8` 抽方法部分。读完后如果某篇的“新问题”需要确认先例，我再用网页搜索补背景。
./reviews/deep_25.md:111:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_25.md:126:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_25.md:136:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_25.md:145:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_25.md:151:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_25.md:166:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_25.md:176:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_25.md:178:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_25.md:190:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_25.md:206:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_25.md:357:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_25.md:378:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_25.md:394:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_25.md:452:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_25.md:453:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_25.md:506:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_25.md:519:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_25.md:546:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_25.md:553:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_25.md:622:/bin/zsh -lc "pdftotext -f 1 -l 8 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:676:as leveraging additional text modalities [7], [8] or infrared
./reviews/deep_25.md:724:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:844:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:879:subspaces guided by text descriptions.
./reviews/deep_25.md:953:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:1266:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:1408:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:1543:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:1812:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:1982:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:1987:/bin/zsh -lc "pdftotext -f 1 -l 8 'Spectrum-guided feature enhancement network for event person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:2019:Event contextual semantics.
./reviews/deep_25.md:2042:incorporate substantial noise, as depicted in Fig. 1. This makes it challenging to directly adapt Re-ID technology from RGB-based to eventbased modalities. However, rich contextual semantics from RGB modalities remain critical for object perception and recognition. To address
./reviews/deep_25.md:2048:more computational resources and imposes stricter data quality requirements. In this context, Ahmad et al. [5] pioneered person matching using
./reviews/deep_25.md:2067:0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_25.md:2140:resolution gaps, while alternative paradigms like text-aided retrieval
./reviews/deep_25.md:2141:in [21,22] leverage textual descriptions for zero-shot person retrieval,
./reviews/deep_25.md:2227:generation results in insuﬃcient contextual aggregation; (3) existing
./reviews/deep_25.md:2274:we attempt to employ an attention mechanism for global semantic interaction to strengthen contextual semantic representations. To achieve
./reviews/deep_25.md:2338:and the loss of texture details. Moreover, the sparsity of event semantics results in weak interactions among semantics and insuﬃcient contextual semantic information. The contextual semantic is detrimental
./reviews/deep_25.md:2339:to the model’s ability to perceive and discriminate semantics. In response, we attempt to employ an attention strategy to achieve interaction and fusion among event semantics, thereby enhancing the representation of event contextual semantics. We ﬁrst compute aﬃnities
./reviews/deep_25.md:2477:richness between the two modalities makes it challenging for RGB models to perceive and capture suﬃcient contextual semantics from events.
./reviews/deep_25.md:2856:SIM further enriches the structured contextual semantics through interactions
./reviews/deep_25.md:3012:we use Spatial Interaction Enhancement (SIM), and the person semantics become more coherent and the contextual semantics are enhanced.
./reviews/deep_25.md:3014:local semantic element and all semantic elements. Correspondingly, connections can be established among diﬀerent semantics to enhance contextual semantic representation. Accordingly, we can clearly observe
./reviews/deep_25.md:3035:severely impair the model’s perception of contextual semantics and its
./reviews/deep_25.md:3042:/bin/zsh -lc "pdftotext -f 1 -l 8 'Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:3125:0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_25.md:4184:Comparison with state-of-the-art methods in the context of imbalance settings on the SYSU-MM01 and RegDB datasets.
./reviews/deep_25.md:4592:context suggests that while our approach is eﬀective, its full potential
./reviews/deep_25.md:4976:/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:4986:degradation or limited alignment capacity. Recent vision-language models leverage textual semantics for
./reviews/deep_25.md:4989:modality-agnostic textual descriptions from images as alignment cues. Specifically, we design a dual-branch
./reviews/deep_25.md:4990:Text Semantic Generation Module that includes: (1) an Attribute-Aware text description Generation module
./reviews/deep_25.md:4991:using prompt-based templates to capture modality-invariant identity cues, and (2) a Contour-Aware text
./reviews/deep_25.md:4992:prompt Module that provides complementary structural information often missing in textual descriptions.
./reviews/deep_25.md:4995:construct category-level textual prototypes as global semantic anchors to enhance cross-modal consistency.
./reviews/deep_25.md:5047:implicit textual semantics derived from image features as auxiliary alignment signals. However,
./reviews/deep_25.md:5049:textual information. Such token representations lack sufficient semantic richness and generalization
./reviews/deep_25.md:5052:incorporates an image-to-text generation paradigm to construct discriminative and modalityinvariant textual descriptions, thereby replacing traditional learnable token vectors. The proposed
./reviews/deep_25.md:5053:framework consists of two complementary modules: (1) an attribute-aware text description
./reviews/deep_25.md:5054:generation (ATG), which synthesizes identity-specific textual descriptions based on pre-defined
./reviews/deep_25.md:5056:information such as color, and (2) a contour-aware text prompt (CTP) module, which captures
./reviews/deep_25.md:5062:representations in the form of learnable textual vectors, making their direct integration nontrivial. To
./reviews/deep_25.md:5063:bridge this gap, we introduce a text re-definition module (TRM) that serves two key functions:
./reviews/deep_25.md:5065:enriched textual prototype and (2) further narrowing the modality gap between generated textual
./reviews/deep_25.md:5066:and visual features via adaptive refinement. The final class-level textual prototypes derived from
./reviews/deep_25.md:5080:(b) “CSDN” [25], a CLIP-based VI-ReID variant, uses learnable feature vectors to represent textual semantics,
./reviews/deep_25.md:5081:providing alignment cues for cross-modal matching. (c) Our proposed method introduces an image-to-text
./reviews/deep_25.md:5082:generator to produce rich, modality-invariant textual descriptions for images from different modalities, which
./reviews/deep_25.md:5085:— Modality-invariant text semantic generation: We introduce a dual-branch text generation
./reviews/deep_25.md:5090:— Unified text representation and fine-grained alignment: A TRM is proposed to integrate heterogeneous semantic descriptions into a unified representation. This unified text is further aligned
./reviews/deep_25.md:5093:— Prototype-guided semantic alignment: We derive category-level textual prototypes from the
./reviews/deep_25.md:5137:Recent advances in vision-language pretraining [16], image–text retrieval [13], and video captioning
./reviews/deep_25.md:5139:person ReID, especially in text–image-based scenarios [8, 21, 27]. These methods typically aim to
./reviews/deep_25.md:5140:enhance retrieval performance by introducing strong semantic alignment cues between text and
./reviews/deep_25.md:5142:significant challenges, including the lack of suitable textual supervision and the substantial modality
./reviews/deep_25.md:5144:prompt learning [26], which transforms identity labels into learnable text embeddings that serve
./reviews/deep_25.md:5145:as semantic supervision. These textual embeddings guide the visual encoder during training,
./reviews/deep_25.md:5150:optimize visible-infrared alignment (VIA) and visual-text reasoning (VTR), improving both
./reviews/deep_25.md:5162:text modalities remains underexplored. These challenges motivate our method, which introduces
./reviews/deep_25.md:5163:structured textual representations that are both discriminative and modality-invariant, effectively
./reviews/deep_25.md:5176:of identity classes, our framework introduces a text-enhanced cross-modal alignment strategy to
./reviews/deep_25.md:5178:To exploit semantic priors, we first leverage a VLM to generate image-conditioned textual
./reviews/deep_25.md:5189:that constructs contour-level textual descriptions T𝑠 = {𝑡𝑖𝑠 }𝐶𝑖=1 for each identity class. Unlike
./reviews/deep_25.md:5192:To effectively integrate both instance-level and class-level textual semantics, we introduce a TRM,
./reviews/deep_25.md:5194:which fuses 𝑡𝑖𝑣 , 𝑡𝑖𝑟 , and 𝑡 𝑦𝑠 𝑖 into unified, identity-sensitive textual representations T𝑜 = {𝑡𝑖𝑣𝑜 , 𝑡𝑖𝑟𝑜 }𝑖=1
./reviews/deep_25.md:5197:Discriminative Feature Set T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 by aggregating the class centers of the redefined textual
./reviews/deep_25.md:5199:features across modalities. By enforcing cross-modal features to align with corresponding text
./reviews/deep_25.md:5209:attribute-based text generation scheme that converts visual attributes into formalized textual
./reviews/deep_25.md:5229:descriptions following the template 𝜏, yielding textual representations 𝑡𝑖𝑣 and 𝑡𝑖𝑟 for the 𝑖th visible and
./reviews/deep_25.md:5236:anchors for visual–textual alignment across modalities.
./reviews/deep_25.md:5238:As a result, we construct an attribute-enhanced textual dataset T = {𝑡𝑖𝑣 , 𝑡𝑖𝑟 }𝑖=1
./reviews/deep_25.md:5240:visual–textual alignment but also introduces human-interpretable and modality-agnostic cues into
./reviews/deep_25.md:5246:While the attribute-aware textual descriptions provide valuable cues for pedestrian recognition,
./reviews/deep_25.md:5267:To enable textual modeling of sketch semantics, we adopt a learnable prompt tuning strategy
./reviews/deep_25.md:5268:inspired by CLIP-ReID [4]. We define an identity-related textual template: 𝒕𝑖𝑠 = “A photo of a
./reviews/deep_25.md:5273:the visual encoder 𝐸 𝑣 and the text encoder 𝐸𝑡 to obtain feature embeddings:
./reviews/deep_25.md:5281:image–text alignment objective consisting of an image-to-text loss and a text-to-image loss:
./reviews/deep_25.md:5333:Through the above procedure, we obtain a set of Contour-aware textual features T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 ,
./reviews/deep_25.md:5340:Although the instantiated attribute texts are generated per image and the Contour-aware prompts
./reviews/deep_25.md:5341:are constructed at the class level, the inherent semantic gap between these two types of textual
./reviews/deep_25.md:5344:contour cues into a unified and discriminative textual feature space.
./reviews/deep_25.md:5346:representation 𝒇𝑖 using the visual encoder. In parallel, we feed its attribute-based textual description
./reviews/deep_25.md:5347:𝒕𝑖 and the corresponding sketch-based class textual feature 𝒕𝑖𝑠 into a textual fusion network 𝐸𝑎
./reviews/deep_25.md:5348:composed of three fully connected layers to obtain an initial fused textual representation:
./reviews/deep_25.md:5357:the fused text 𝒇𝑖𝑎 as input to generate a redefined, image-aware textual representation 𝒇𝑖 :
./reviews/deep_25.md:5377:semantic priors of the original textual input.
./reviews/deep_25.md:5379:To ensure that the redefined textual features 𝒇𝑖 are both semantically aligned with the visual content and discriminative with respect to pedestrian identity, we employ a combination of contrastive
./reviews/deep_25.md:5392:where the image-to-text and text-to-image contrastive losses are defined as:
./reviews/deep_25.md:5459:training, we freeze the parameters of the visual encoder and text encoder and update only the
./reviews/deep_25.md:5462:After convergence, the redefined feature 𝒇𝑖 serves as an enriched and interaction-aware textual
./reviews/deep_25.md:5468:In this section, we present our proposed model, text-guided attribute and prototype alignment
./reviews/deep_25.md:5481:textual cues from images as semantic guidance. In particular, we introduce a Text Prototype
./reviews/deep_25.md:5482:Discriminative Loss, which encourages visual features to align with class-level text prototypes that
./reviews/deep_25.md:5494:where identity loss L𝑖𝑑 , weighted regularized triplet loss L𝑤𝑟𝑡 , text prototype discriminative loss
./reviews/deep_25.md:5517:/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-to-image Person Search based on Semantic Reorganization.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:5547:Despite significant advancements in text-based person search (TBPS), challenges remain in acquiring large-scale
./reviews/deep_25.md:5549:modality gaps between textual and visual semantics lead to a degradation in retrieval performance. To
./reviews/deep_25.md:5550:address these challenges, we propose a novel framework, i.e., text-to-image Person Search based on Semantic
./reviews/deep_25.md:5553:utilizes a large multimodal language model to generate corresponding textual descriptions, thereby increasing
./reviews/deep_25.md:5556:given text feature and fusing them into a whole. This approach alleviates the negative impact of modality
./reviews/deep_25.md:5558:text-based person search datasets. Source code is at https://github.com/JielongHe/PSSR.
./reviews/deep_25.md:5564:based on textual descriptions. Its goal is to establish a semantic link
./reviews/deep_25.md:5570:other text–image retrieval tasks (Lülf et al., 2024), TBPS requires more
./reviews/deep_25.md:5571:detailed textual descriptions of pedestrian images. As a result, data
./reviews/deep_25.md:5572:annotation for TBPS is more complex than for text–image retrieval. And
./reviews/deep_25.md:5577:differences, making text-to-image retrieval more difficult than imageto-image retrieval. Therefore, effective measures must be taken to
./reviews/deep_25.md:5584:(MLLM) to generate textual descriptions. The generated image–text
./reviews/deep_25.md:5588:and demands high-quality textual descriptions. To effectively augment
./reviews/deep_25.md:5590:(2024) used large language models to diversify and rewrite text in the
./reviews/deep_25.md:5595:new images. They also modified the original textual descriptions using
./reviews/deep_25.md:5614:1077-3142/© 2026 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_25.md:5620:Fig. 1. Comparison of different data augmentation methods. (a) Generate images using diffusion models based on attributes, while generating text based on the
./reviews/deep_25.md:5621:LLM. (b) Image–text pair augmentation based on semantic information and pedestrian image structure information.
./reviews/deep_25.md:5636:consistency between the token positions in the text description and the
./reviews/deep_25.md:5639:two texts. Accordingly, we utilize the MLLM model to produce textual
./reviews/deep_25.md:5641:accuracy of the textual descriptions.
./reviews/deep_25.md:5644:align images and texts in a joint embedding space and use crossmodal matching loss functions to align image–text pairs. Common loss
./reviews/deep_25.md:5648:making it widely used in text-to-image person retrieval. However, due
./reviews/deep_25.md:5653:by minimizing the KL divergence between the normalized image–text
./reviews/deep_25.md:5659:features extracted from MLLM as anchors to align visual and text features more closely. Lu et al. (2024) proposed multimodality contrastive
./reviews/deep_25.md:5663:between text and visual data. On the other hand, some work attempts
./reviews/deep_25.md:5670:the same modality (e.g., traditional Re-ID image retrieval) but is inherently not satisfied in cross-modal (text–image) scenarios. Meanwhile,
./reviews/deep_25.md:5672:that the average cosine distance between image–image pairs is significantly smaller than that between text–image pairs, with the latter
./reviews/deep_25.md:5674:that text lacks stable local structural support in feature space. These
./reviews/deep_25.md:5676:effectively utilize textual semantics in cross-modal retrieval. Based on
./reviews/deep_25.md:5678:Mechanism (CSRM) during the testing phase to build a more semantically reliable cross-modal neighborhood structure for text. Through
./reviews/deep_25.md:5681:the text semantics and also representative in image space. We then use
./reviews/deep_25.md:5683:new text query features. Compared to methods that improve model
./reviews/deep_25.md:5704:expanded training data, the alignment between the textual and visual
./reviews/deep_25.md:5707:selection and Jaccard weight fusion, enabling the query text to adjust
./reviews/deep_25.md:5717:text and visual representations remains a core challenge. Li et al.
./reviews/deep_25.md:5720:textual features, thereby establishing the groundwork for advanced
./reviews/deep_25.md:5733:concentrate on key local features in both images and text descriptions,
./reviews/deep_25.md:5741:extracts co-saliency features between images and text, enhancing the
./reviews/deep_25.md:5749:retrieval settings, Y. Lu et al. (2025) explored an interactive text-based
./reviews/deep_25.md:5755:appropriate image features to represent the counterpart text, thereby
./reviews/deep_25.md:5765:the corresponding text feature, thereby mitigating the negative
./reviews/deep_25.md:5768:and 51.77% on the CUHK-PEDES, ICFG-PEDES and RSTPReid,
./reviews/deep_25.md:5782:In TBPS, both image augmentation and generation of semantically consistent image–text pairs are crucial. Z. Li et al. (2024) leveraged large language models to rewrite textual descriptions in existing
./reviews/deep_25.md:5783:datasets, generating diverse text, though only addressing textual augmentation. The MALS dataset (Yang et al., 2023) employs diffusion
./reviews/deep_25.md:5784:models and text calibration to generate large-scale synthetic data. Tan
./reviews/deep_25.md:5785:et al. (2024) used the LUPerson dataset (Fu et al., 2021) and Multimodal Large Language Models (MLLM) to generate textual descriptions
./reviews/deep_25.md:5787:large language models for textual annotations, mitigating privacy and
./reviews/deep_25.md:5792:image structure and produce corresponding text by fine-tuning the
./reviews/deep_25.md:5810:images. Finally, the generated image–text pairs { ̂
./reviews/deep_25.md:5822:In order to better map images and text description to the common feature space, we designed a Clip-based dual-branch network
./reviews/deep_25.md:5824:consists of two unimodal encoders for image and text encoding, respectively. For a given image–text pair (𝐼, 𝑇 ), the image encoder first
./reviews/deep_25.md:5831:Similarly, the text 𝑇 undergoes processing by the text encoder, yielding
./reviews/deep_25.md:5833:a series of textual representations 𝐹 𝑡 = {𝑓1𝑡 , … , 𝑓𝑁
./reviews/deep_25.md:5836:denotes the global textual representation.
./reviews/deep_25.md:5842:image and text representations by minimizing the discrepancy between
./reviews/deep_25.md:5844:mini-batch of 𝑁 image–text pairs, the matching probability 𝑝𝑖,𝑗 is
./reviews/deep_25.md:5847:In order to enhance the diversity of image–text pairs, we propose a
./reviews/deep_25.md:5863:textual descriptions, we fine-tune the BLIP (Bootstrapping Language
./reviews/deep_25.md:5864:Image Pretraining) model using the original image–text pairs. The BLIP
./reviews/deep_25.md:5865:model generates textual descriptions based on image content. After
./reviews/deep_25.md:5868:the first three words of the corresponding upper body text as the
./reviews/deep_25.md:5869:prompt, for example, ‘‘A man in’’, and input this prompt into the finetuned BLIP model to generate a new textual description. Specifically,
./reviews/deep_25.md:5870:̂ and the corresponding upper body text be
./reviews/deep_25.md:5874:BLIP model to obtain the corresponding generated text 𝑇̂ :
./reviews/deep_25.md:5893:distribution. The image-to-text SDM loss 𝐿𝑖2𝑡 is then computed using
./reviews/deep_25.md:5914:This text 𝑇̂ serves as the textual description for the new image, further enhancing the diversity and semantic reliability of the generated
./reviews/deep_25.md:5928:is a small number to avoid numerical issues. Symmetrically, the textto-image SDM loss 𝐿𝑡2𝑖 is calculated by exchanging the image and text
./reviews/deep_25.md:5936:the image-to-text and text-to-image losses:
./reviews/deep_25.md:5939:similarity between the text feature 𝑡𝑚 and the image 𝑣𝑖 , defined as:
./reviews/deep_25.md:5949:negative samples within the same modality, thereby enhancing intramodal discrimination. Since the text is generated by a Multi-modal
./reviews/deep_25.md:5976:CUHK-PEDES (Li et al., 2017) dataset, a pioneer in text-based person retrieval, includes 34,054/68,108 images/sentences for 11,003
./reviews/deep_25.md:5979:text pairs representing 3,102/1,000 identities. The RSTPReid (Zhu
./reviews/deep_25.md:5991:image features to represent text features and use the fused features for
./reviews/deep_25.md:5993:the query text feature 𝑡𝑚 and all gallery images {𝑣𝑖 }𝐺
./reviews/deep_25.md:6017:similarity between 𝑣𝑖 and all existing textual representations and obtain
./reviews/deep_25.md:6018:the set of k-nearest texts of 𝑣𝑙 as 𝑁𝑘 (𝑣𝑖 ) = {𝑡1 , 𝑡2 , … , 𝑡𝑘 }. Therefore,
./reviews/deep_25.md:6020:set of cross-modal k-reciprocal nearest images 𝑅𝑘 (𝑡𝑚 ) of query text
./reviews/deep_25.md:6034:respectively. Finally, on the RSTPReid dataset, our method achieved
./reviews/deep_25.md:6037:In order to better fuse the text 𝑡𝑚 with its cross-modal k-reciprocal
./reviews/deep_25.md:6048:At this point, the fused text feature 𝑡𝑓𝑚𝑢𝑠𝑒𝑑 is used as a new query
./reviews/deep_25.md:6049:text feature for image retrieval, which is able to capture cross-modal
./reviews/deep_25.md:6050:inter-modal correlations more accurately than the original text feature
./reviews/deep_25.md:6062:On this basis, we enhance the fusion effect between textual features
./reviews/deep_25.md:6066:of the query text feature 𝑡𝑚 itself. A weighting strategy is introduced,
./reviews/deep_25.md:6119:RSTPReid
./reviews/deep_25.md:6130:IRRA (Jiang and Ye, 2023)
./reviews/deep_25.md:6135:IRRA+DP (Z. Song et al., 2024)
./reviews/deep_25.md:6267:RSTPReid
./reviews/deep_25.md:6334:the diversity of the training dataset. Unlike the ‘‘IRRA+DP’’ method,
./reviews/deep_25.md:6337:pedestrian images. Although the ‘‘IRRA+DP’’ does generate more natural and realistic images, visual quality does not necessarily equate to
./reviews/deep_25.md:6338:model training effectiveness. ‘‘IRRA+DP’’ relies on the diffusion model
./reviews/deep_25.md:6341:the generated images and the textual semantics, and the diversity of
./reviews/deep_25.md:6350:‘‘IRRA+DP’’ are more natural, from the perspective of the cross-modal
./reviews/deep_25.md:6361:of modality discrepancies on retrieval performance by selecting appropriate image features to fuse with text features during the testing phase.
./reviews/deep_25.md:6370:with 𝑡𝑒𝑥𝑡_𝑡𝑟𝑖𝑝𝑙𝑒𝑡 ’ denotes the inclusion of triplet loss within the text
./reviews/deep_25.md:6374:a performance drop because the text generated by the MLLM model
./reviews/deep_25.md:6393:and RSTPReid datasets increased by 4.94%, 7.63%, and 1.53%, respectively. When CSRM was introduced individually into the baseline,
./reviews/deep_25.md:6398:the image and text modalities, resulting in the traditional intra-modal
./reviews/deep_25.md:6402:images and fuse them with the corresponding the text. This approach
./reviews/deep_25.md:6430:RSTPReid
./reviews/deep_25.md:6486:RSTPReid
./reviews/deep_25.md:6526:in the text description. The third and fifth column negative samples exhibit the muted plaid jacket, while the fourth column negative sample
./reviews/deep_25.md:6531:the incomplete nature of the textual descriptions.
./reviews/deep_25.md:6534:triangles represent text queries, circles represent image samples. The
./reviews/deep_25.md:6536:indicate belonging to different IDs. Without CSRM (Fig. 4(a)), the text
./reviews/deep_25.md:6541:that CSRM really helps the model focus on the right text–image pairs.
./reviews/deep_25.md:6546:where the neighborhood images do not match the original query text.
./reviews/deep_25.md:6576:images that best align with the semantic content of the text query. This
./reviews/deep_25.md:6587:visually differ greatly, they are still quite similar to the text description.
./reviews/deep_25.md:6594:Fig. 4. Comparison of T-SNE visualization on the CSRM. Triangles represent query text, circles represent gallery images, and colors represent IDs.
./reviews/deep_25.md:6596:Fig. 5. Comparison of T-SNE visualization on the Baseline and our PSSR. Triangles represent text, circles represent images, and colors represent categories.
./reviews/deep_25.md:6690:measurements between text and image in local regions is limited.
./reviews/deep_25.md:6706:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:6745:pedestrian occlusion, a Texture-Aware Decoder is introduced to perceive texture features and leverage their
./reviews/deep_25.md:6785:0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_25.md:6801:ﬁltered by PPM as constraints, and learns discriminative textures hierarchically through multi-head attention and cross-attention mechanisms.
./reviews/deep_25.md:6802:Then, the discriminative textures are leveraged to enhance the eﬀective semantic representation of relevant regions, serving as a signiﬁcant
./reviews/deep_25.md:6823:• A texture-aware decoder is designed that enhances the semantic representation of the eﬀective regions by hierarchically learning discriminative textures, serving as an important distinction for diﬀerent
./reviews/deep_25.md:6841:for texture features through pose features, while texture features supplement pose features with ﬁne-grained discriminative clues. The combination of the two forms dual constraints of structure and detail. Speciﬁcally, instead of performing pose-to-global mapping, observing that the
./reviews/deep_25.md:6852:non-targets. As the appearance of target and non-target pedestrians often exhibits notable texture variations, it is more feasible to leverage
./reviews/deep_25.md:6853:these texture variations to enhance the representation of target regions
./reviews/deep_25.md:6855:appearance. Therefore, we employ textures to augment the representation of discriminative appearance, thereby increasing the inter-class
./reviews/deep_25.md:6931:To address this issue, we introduce a texture-aware decoder consisting
./reviews/deep_25.md:6932:of a texture-extraction layer and a context-extraction layer. The former
./reviews/deep_25.md:6933:is designed to capture the contextual feature of the original sequence
./reviews/deep_25.md:6934:0 , while the latter applies cross-attention between the contextual features and patch features to extract distinctive textures. The structure of
./reviews/deep_25.md:6957:non-occluded semantics. In PFD, pose information is directly embedded into both global contextual features and local features. Although a
./reviews/deep_25.md:6961:pose-patch mapping and introduce a texture-aware decoder to realize
./reviews/deep_25.md:7013:through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.
./reviews/deep_25.md:7164:diﬀerent from the traditional transformer architecture, our TAD comprises two types of layers: the context extraction layer is based on a
./reviews/deep_25.md:7165:multi-head attention mechanism, which captures contextual dependencies of textures from diﬀerent perspectives through parallel multiple attention heads, and adaptively aggregates global semantic information.
./reviews/deep_25.md:7166:The texture extraction layer utilizes a cross attention mechanism to establish a unidirectional guidance channel between texture and contextual features. By dynamically ﬁltering and enhancing key texture regions
./reviews/deep_25.md:7167:based on contextual semantics and pose features, it enables the learning of more discriminative texture appearance features. The detailed
./reviews/deep_25.md:7174:contextual features 𝐹𝑐𝑁 are used as the input of TELs to assist the mutual information attention from appearance to texture. Similar to CEL,
./reviews/deep_25.md:7183:cosine similarity between the obtained texture-aware feature 𝐹𝑡 and the
./reviews/deep_25.md:7185:locations. TAD leverages the continuity of texture and contextual features to further augment the target features that are corrupted by other
./reviews/deep_25.md:7313:after texture augmentation. Next, we provide a detailed introduction to
./reviews/deep_25.md:7333:3.4.1. Context extraction layer
./reviews/deep_25.md:7359:is the (𝑖 − 1)th context-extraction feature. 𝐼𝑁(⋅) represent
./reviews/deep_25.md:7379:After 𝑁 layers of CELs, the contextual features 𝐹𝑐𝑁 can be output and
./reviews/deep_25.md:7380:fed into each TEL for disentangled learning from appearance to texture.
./reviews/deep_25.md:7416:mapping between pixels. Subsequently, with the assistance of pose mapping, 𝐹̃𝑡𝑖 is used as the value input into MHCA to achieve texture reﬁnement of 𝐹̂ 𝑖 .
./reviews/deep_25.md:7439:In this way, 𝐹̂𝑡𝑖 gradually carries more texture information, promoting TELs to generate accurate and detailed texture features. Similarly,
./reviews/deep_25.md:7441:texture capture process is as follows:
./reviews/deep_25.md:7448:and texture augmented feature (in Eq. (20)), respectively.
./reviews/deep_25.md:7463:After 𝑁 layers of TELs, the texture-aware feature 𝐹𝑡 is collected. In
./reviews/deep_25.md:7465:precise and ﬁne-grained textures.
./reviews/deep_25.md:7517:Decoder Loss: To more eﬀectively guide the encoder in capturing ﬁnegrained texture semantics, the encoding loss 𝑑𝑒 is employed. It is used
./reviews/deep_25.md:7725:to generate simulated data. Instead, it only focuses on the target individual, which includes the human skeleton and ﬁne-grained texture
./reviews/deep_25.md:7901:Contribution Analysis of TAD Components on OccludedDuke (%). TELs and CELs denote Texture extraction layers and Context extraction layers in TAD. Bold indicates
./reviews/deep_25.md:8151:and 4.2 %/4.6 % in Rank-1/mAP, respectively. TAD leverages the continuity of texture features to further enhance the target features that
./reviews/deep_25.md:8155:texture-aware encoder (left column), TTPM accurately re-identiﬁes the
./reviews/deep_25.md:8157:the third row on the right) with the aid of texture-aware encoder. Additionally, we present some failure cases, as shown in Fig. 6. Although
./reviews/deep_25.md:8160:textures. This result indicates that the proposed TTPM still has limitations in relying on texture features. While these features enable accurate
./reviews/deep_25.md:8198:/bin/zsh -lc "pdftotext -f 1 -l 8 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | rg -n \""'^(III|IV|A'"\\.|B\\.|C\\.|D\\.|[0-9]+\\)|Methodology|Overview|Structured|Modality|Loss|Experiments|Datasets|Implementation)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:8200:/bin/zsh -lc "pdftotext -f 1 -l 8 'Spectrum-guided feature enhancement network for event person re-identification.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|III|IV|Method|Experiment|MSAM|CPDM|Loss|Contribution|Discussion)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:8391:/bin/zsh -lc "pdftotext -f 1 -l 8 'Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Experiment|Stochastic|Cross|Class|Ablation|Which|Effectiveness)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:9146:/bin/zsh -lc "pdftotext -f 1 -l 8 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | sed -n '306,1341p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:9172:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:9485:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:9627:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:9762:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:10031:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:10186:/bin/zsh -lc "pdftotext -f 1 -l 8 'Spectrum-guided feature enhancement network for event person re-identification.pdf' - | sed -n '245,504p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:10229:we attempt to employ an attention mechanism for global semantic interaction to strengthen contextual semantic representations. To achieve
./reviews/deep_25.md:10293:and the loss of texture details. Moreover, the sparsity of event semantics results in weak interactions among semantics and insuﬃcient contextual semantic information. The contextual semantic is detrimental
./reviews/deep_25.md:10294:to the model’s ability to perceive and discriminate semantics. In response, we attempt to employ an attention strategy to achieve interaction and fusion among event semantics, thereby enhancing the representation of event contextual semantics. We ﬁrst compute aﬃnities
./reviews/deep_25.md:10432:richness between the two modalities makes it challenging for RGB models to perceive and capture suﬃcient contextual semantics from events.
./reviews/deep_25.md:10450:/bin/zsh -lc "pdftotext -f 1 -l 8 'Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf' - | sed -n '166,505p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:10806:/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Framework|ATG|Contour|TRM|Text-Guided|Experimental|Ablation|Prototype|Loss)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:10829:/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-to-image Person Search based on Semantic Reorganization.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Semantic|Cross|Training|Ablation|Comparison|Analysis)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:11002:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Overview|Multi|Pose|Texture|Loss|Ablation|Effectiveness|Comparison)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:11018:626:3.4.1. Context extraction layer
./reviews/deep_25.md:11274:/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '188,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:11287:of identity classes, our framework introduces a text-enhanced cross-modal alignment strategy to
./reviews/deep_25.md:11289:To exploit semantic priors, we first leverage a VLM to generate image-conditioned textual
./reviews/deep_25.md:11300:that constructs contour-level textual descriptions T𝑠 = {𝑡𝑖𝑠 }𝐶𝑖=1 for each identity class. Unlike
./reviews/deep_25.md:11303:To effectively integrate both instance-level and class-level textual semantics, we introduce a TRM,
./reviews/deep_25.md:11305:which fuses 𝑡𝑖𝑣 , 𝑡𝑖𝑟 , and 𝑡 𝑦𝑠 𝑖 into unified, identity-sensitive textual representations T𝑜 = {𝑡𝑖𝑣𝑜 , 𝑡𝑖𝑟𝑜 }𝑖=1
./reviews/deep_25.md:11308:Discriminative Feature Set T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 by aggregating the class centers of the redefined textual
./reviews/deep_25.md:11310:features across modalities. By enforcing cross-modal features to align with corresponding text
./reviews/deep_25.md:11320:attribute-based text generation scheme that converts visual attributes into formalized textual
./reviews/deep_25.md:11340:descriptions following the template 𝜏, yielding textual representations 𝑡𝑖𝑣 and 𝑡𝑖𝑟 for the 𝑖th visible and
./reviews/deep_25.md:11347:anchors for visual–textual alignment across modalities.
./reviews/deep_25.md:11349:As a result, we construct an attribute-enhanced textual dataset T = {𝑡𝑖𝑣 , 𝑡𝑖𝑟 }𝑖=1
./reviews/deep_25.md:11351:visual–textual alignment but also introduces human-interpretable and modality-agnostic cues into
./reviews/deep_25.md:11357:While the attribute-aware textual descriptions provide valuable cues for pedestrian recognition,
./reviews/deep_25.md:11378:To enable textual modeling of sketch semantics, we adopt a learnable prompt tuning strategy
./reviews/deep_25.md:11379:inspired by CLIP-ReID [4]. We define an identity-related textual template: 𝒕𝑖𝑠 = “A photo of a
./reviews/deep_25.md:11384:the visual encoder 𝐸 𝑣 and the text encoder 𝐸𝑡 to obtain feature embeddings:
./reviews/deep_25.md:11392:image–text alignment objective consisting of an image-to-text loss and a text-to-image loss:
./reviews/deep_25.md:11444:Through the above procedure, we obtain a set of Contour-aware textual features T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 ,
./reviews/deep_25.md:11451:Although the instantiated attribute texts are generated per image and the Contour-aware prompts
./reviews/deep_25.md:11452:are constructed at the class level, the inherent semantic gap between these two types of textual
./reviews/deep_25.md:11455:contour cues into a unified and discriminative textual feature space.
./reviews/deep_25.md:11457:representation 𝒇𝑖 using the visual encoder. In parallel, we feed its attribute-based textual description
./reviews/deep_25.md:11458:𝒕𝑖 and the corresponding sketch-based class textual feature 𝒕𝑖𝑠 into a textual fusion network 𝐸𝑎
./reviews/deep_25.md:11459:composed of three fully connected layers to obtain an initial fused textual representation:
./reviews/deep_25.md:11468:the fused text 𝒇𝑖𝑎 as input to generate a redefined, image-aware textual representation 𝒇𝑖 :
./reviews/deep_25.md:11488:semantic priors of the original textual input.
./reviews/deep_25.md:11490:To ensure that the redefined textual features 𝒇𝑖 are both semantically aligned with the visual content and discriminative with respect to pedestrian identity, we employ a combination of contrastive
./reviews/deep_25.md:11503:where the image-to-text and text-to-image contrastive losses are defined as:
./reviews/deep_25.md:11570:training, we freeze the parameters of the visual encoder and text encoder and update only the
./reviews/deep_25.md:11573:After convergence, the redefined feature 𝒇𝑖 serves as an enriched and interaction-aware textual
./reviews/deep_25.md:11579:In this section, we present our proposed model, text-guided attribute and prototype alignment
./reviews/deep_25.md:11592:textual cues from images as semantic guidance. In particular, we introduce a Text Prototype
./reviews/deep_25.md:11593:Discriminative Loss, which encourages visual features to align with class-level text prototypes that
./reviews/deep_25.md:11605:where identity loss L𝑖𝑑 , weighted regularized triplet loss L𝑤𝑟𝑡 , text prototype discriminative loss
./reviews/deep_25.md:11630:/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-to-image Person Search based on Semantic Reorganization.pdf' - | sed -n '278,505p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:11646:images. Finally, the generated image–text pairs { ̂
./reviews/deep_25.md:11658:In order to better map images and text description to the common feature space, we designed a Clip-based dual-branch network
./reviews/deep_25.md:11660:consists of two unimodal encoders for image and text encoding, respectively. For a given image–text pair (𝐼, 𝑇 ), the image encoder first
./reviews/deep_25.md:11667:Similarly, the text 𝑇 undergoes processing by the text encoder, yielding
./reviews/deep_25.md:11669:a series of textual representations 𝐹 𝑡 = {𝑓1𝑡 , … , 𝑓𝑁
./reviews/deep_25.md:11672:denotes the global textual representation.
./reviews/deep_25.md:11678:image and text representations by minimizing the discrepancy between
./reviews/deep_25.md:11680:mini-batch of 𝑁 image–text pairs, the matching probability 𝑝𝑖,𝑗 is
./reviews/deep_25.md:11683:In order to enhance the diversity of image–text pairs, we propose a
./reviews/deep_25.md:11699:textual descriptions, we fine-tune the BLIP (Bootstrapping Language
./reviews/deep_25.md:11700:Image Pretraining) model using the original image–text pairs. The BLIP
./reviews/deep_25.md:11701:model generates textual descriptions based on image content. After
./reviews/deep_25.md:11704:the first three words of the corresponding upper body text as the
./reviews/deep_25.md:11705:prompt, for example, ‘‘A man in’’, and input this prompt into the finetuned BLIP model to generate a new textual description. Specifically,
./reviews/deep_25.md:11706:̂ and the corresponding upper body text be
./reviews/deep_25.md:11710:BLIP model to obtain the corresponding generated text 𝑇̂ :
./reviews/deep_25.md:11729:distribution. The image-to-text SDM loss 𝐿𝑖2𝑡 is then computed using
./reviews/deep_25.md:11750:This text 𝑇̂ serves as the textual description for the new image, further enhancing the diversity and semantic reliability of the generated
./reviews/deep_25.md:11764:is a small number to avoid numerical issues. Symmetrically, the textto-image SDM loss 𝐿𝑡2𝑖 is calculated by exchanging the image and text
./reviews/deep_25.md:11772:the image-to-text and text-to-image losses:
./reviews/deep_25.md:11775:similarity between the text feature 𝑡𝑚 and the image 𝑣𝑖 , defined as:
./reviews/deep_25.md:11785:negative samples within the same modality, thereby enhancing intramodal discrimination. Since the text is generated by a Multi-modal
./reviews/deep_25.md:11812:CUHK-PEDES (Li et al., 2017) dataset, a pioneer in text-based person retrieval, includes 34,054/68,108 images/sentences for 11,003
./reviews/deep_25.md:11815:text pairs representing 3,102/1,000 identities. The RSTPReid (Zhu
./reviews/deep_25.md:11827:image features to represent text features and use the fused features for
./reviews/deep_25.md:11829:the query text feature 𝑡𝑚 and all gallery images {𝑣𝑖 }𝐺
./reviews/deep_25.md:11853:similarity between 𝑣𝑖 and all existing textual representations and obtain
./reviews/deep_25.md:11854:the set of k-nearest texts of 𝑣𝑙 as 𝑁𝑘 (𝑣𝑖 ) = {𝑡1 , 𝑡2 , … , 𝑡𝑘 }. Therefore,
./reviews/deep_25.md:11856:set of cross-modal k-reciprocal nearest images 𝑅𝑘 (𝑡𝑚 ) of query text
./reviews/deep_25.md:11862:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | sed -n '293,773p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:11877:through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.
./reviews/deep_25.md:12028:diﬀerent from the traditional transformer architecture, our TAD comprises two types of layers: the context extraction layer is based on a
./reviews/deep_25.md:12029:multi-head attention mechanism, which captures contextual dependencies of textures from diﬀerent perspectives through parallel multiple attention heads, and adaptively aggregates global semantic information.
./reviews/deep_25.md:12030:The texture extraction layer utilizes a cross attention mechanism to establish a unidirectional guidance channel between texture and contextual features. By dynamically ﬁltering and enhancing key texture regions
./reviews/deep_25.md:12031:based on contextual semantics and pose features, it enables the learning of more discriminative texture appearance features. The detailed
./reviews/deep_25.md:12038:contextual features 𝐹𝑐𝑁 are used as the input of TELs to assist the mutual information attention from appearance to texture. Similar to CEL,
./reviews/deep_25.md:12047:cosine similarity between the obtained texture-aware feature 𝐹𝑡 and the
./reviews/deep_25.md:12049:locations. TAD leverages the continuity of texture and contextual features to further augment the target features that are corrupted by other
./reviews/deep_25.md:12177:after texture augmentation. Next, we provide a detailed introduction to
./reviews/deep_25.md:12197:3.4.1. Context extraction layer
./reviews/deep_25.md:12223:is the (𝑖 − 1)th context-extraction feature. 𝐼𝑁(⋅) represent
./reviews/deep_25.md:12243:After 𝑁 layers of CELs, the contextual features 𝐹𝑐𝑁 can be output and
./reviews/deep_25.md:12244:fed into each TEL for disentangled learning from appearance to texture.
./reviews/deep_25.md:12280:mapping between pixels. Subsequently, with the assistance of pose mapping, 𝐹̃𝑡𝑖 is used as the value input into MHCA to achieve texture reﬁnement of 𝐹̂ 𝑖 .
./reviews/deep_25.md:12303:In this way, 𝐹̂𝑡𝑖 gradually carries more texture information, promoting TELs to generate accurate and detailed texture features. Similarly,
./reviews/deep_25.md:12305:texture capture process is as follows:
./reviews/deep_25.md:12312:and texture augmented feature (in Eq. (20)), respectively.
./reviews/deep_25.md:12327:After 𝑁 layers of TELs, the texture-aware feature 𝐹𝑡 is collected. In
./reviews/deep_25.md:12329:precise and ﬁne-grained textures.
./reviews/deep_25.md:12347:TTPM 的页面排版导致方法文本顺序有些乱，刚才截取漏了 MFE 的前半段。我会再补一次该论文从 3.1 到 3.3 的开头，避免把 pose-patch mapping 的逻辑链断掉。
./reviews/deep_25.md:12353:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | sed -n '174,400p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:12405:To address this issue, we introduce a texture-aware decoder consisting
./reviews/deep_25.md:12406:of a texture-extraction layer and a context-extraction layer. The former
./reviews/deep_25.md:12407:is designed to capture the contextual feature of the original sequence
./reviews/deep_25.md:12408:0 , while the latter applies cross-attention between the contextual features and patch features to extract distinctive textures. The structure of
./reviews/deep_25.md:12431:non-occluded semantics. In PFD, pose information is directly embedded into both global contextual features and local features. Although a
./reviews/deep_25.md:12435:pose-patch mapping and introduce a texture-aware decoder to realize
./reviews/deep_25.md:12487:through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.
./reviews/deep_25.md:12622:Creator:         LaTeX with acmart 2025/08/27 v2.16 Typesetting articles for the Association for Computing Machinery and hyperref 2024-10-30 v7.01k Hypertext links for LaTeX
./reviews/deep_25.md:12642:/bin/zsh -lc "pdftotext -f 8 -l 14 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | rg -n \"(Ablation|SD-3DGF|MRIE|ECS|local similarity|LSL|appearance|gait|clothes-changing|Table|Fig\\.|Consistency|without|w/o|Impact|Effect|Contribution|ablation|representative|VCCR|CCVID|CCV|MARS|Rank|mAP)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:12906:/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | rg -n \"(Ablation|ablation|ATG|CTP|TRM|prototype|Prototype|attribute|contour|text|Text|SYSU|RegDB|Table|Figure|Fig\\.|Effect|w/o|without|modal|alignment|Rank|mAP|loss|TAP)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:12912:30:training, we freeze the parameters of the visual encoder and text encoder and update only the
./reviews/deep_25.md:12913:33:After convergence, the redefined feature 𝒇𝑖 serves as an enriched and interaction-aware textual
./reviews/deep_25.md:12916:39:In this section, we present our proposed model, text-guided attribute and prototype alignment
./reviews/deep_25.md:12925:52:textual cues from images as semantic guidance. In particular, we introduce a Text Prototype
./reviews/deep_25.md:12926:53:Discriminative Loss, which encourages visual features to align with class-level text prototypes that
./reviews/deep_25.md:12928:65:where identity loss L𝑖𝑑 , weighted regularized triplet loss L𝑤𝑟𝑡 , text prototype discriminative loss
./reviews/deep_25.md:12930:86:Fig. 2. Our framework introduces a dual-branch text generation module to extract modality-invariant semantic
./reviews/deep_25.md:12931:87:descriptions from pedestrian images. Specifically, an ATG constructs identity-relevant textual descriptions
./reviews/deep_25.md:12933:89:the class level. These two complementary descriptions are fused by a TRM into unified textual prototypes,
./reviews/deep_25.md:12938:145:extract contour-relevant regions, enabling the generation of sketch-based textual representations for robust
./reviews/deep_25.md:12940:148:Fig. 5. TRM merges attribute- and sketch-based texts with image features through fusion and transformer
./reviews/deep_25.md:12941:149:interaction, producing discriminative, image-aware textual representations for each instance.
./reviews/deep_25.md:12942:203:vectors, 𝒇𝑦𝑡𝑖 denotes the class-level textual prototype feature corresponding to the identity label 𝑦𝑖 ,
./reviews/deep_25.md:12944:218:Here, 𝑛𝑞 denotes the number of textual features corresponding to the identity label 𝑦𝑖 .
./reviews/deep_25.md:12946:224:Text Semantic Generation Module. This module generates two types of textual features: on the
./reviews/deep_25.md:12947:225:one hand, a large language model is employed to produce attribute-descriptive texts based on
./reviews/deep_25.md:12950:228:the attribute-aware textual features with the contour-aware semantic features to form a newly
./reviews/deep_25.md:12951:229:defined textual representation. Based on these redefined textual features, we compute class-wise
./reviews/deep_25.md:12965:279:cosine schedule. In the third stage, we train the TRM while freezing both the visual and text
./reviews/deep_25.md:12969:295:without incorporating textual information from the images. We found that our method outperforms
./reviews/deep_25.md:12974:306:about 0.8%. These experimental results indicate that incorporating rich textual information can
./reviews/deep_25.md:12976:309:scenarios, suggesting that the introduced textual information might be subject to specific contextual
./reviews/deep_25.md:12977:312:model-based methods, such as MIP, J-ReID, and CSDN. These methods leverage textual information
./reviews/deep_25.md:12990:585:for constructing text prompts that better focus on pedestrian recognition information, further
./reviews/deep_25.md:12991:586:enhancing performance. This suggests that richer semantic textual prompts can effectively guide
./reviews/deep_25.md:12994:590:such as MUN, perform better than the language model–based method MIP that introduces textual
./reviews/deep_25.md:12996:592:to the fact that the design of the textual prompts in the MIP method is not yet optimal, making it
./reviews/deep_25.md:12997:595:cross-modal person ReID depends on generating effective textual prompts and designing suitable
./reviews/deep_25.md:13007:649:follows: we utilized the textual information generated from ATG to compute the category
./reviews/deep_25.md:13008:650:prototypes of the text as alignment cues, guiding images from different modalities to align
./reviews/deep_25.md:13009:651:towards these cues. As shown in the table, the textual information generated based on image
./reviews/deep_25.md:13012:655:textual information based on image pedestrian attributes is effective.
./reviews/deep_25.md:13016:659:class-level textual features, which are employed as additional alignment cues to facilitate
./reviews/deep_25.md:13019:662:accuracy. These results validate that contour-guided textual semantics provide meaningful
./reviews/deep_25.md:13023:666:class-level textual prototypes are independently generated by ATG and CTP, where ATG encodes high-level semantic attributes and CTP captures local structural contours of pedestrians.
./reviews/deep_25.md:13024:667:These two sets of prototypes are averaged to obtain a unified textual representation, which
./reviews/deep_25.md:13031:674:configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual
./reviews/deep_25.md:13034:TAP-align 的消融说明已经找到了：它把属性文本、轮廓文本、重定义模块分开验证。我要再取一次表 3 周边的具体文字和数值，方便最后不只写“有消融”，而能说明闭环证据是什么。
./reviews/deep_25.md:13048:/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '602,720p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:13097:follows: we utilized the textual information generated from ATG to compute the category
./reviews/deep_25.md:13098:prototypes of the text as alignment cues, guiding images from different modalities to align
./reviews/deep_25.md:13099:towards these cues. As shown in the table, the textual information generated based on image
./reviews/deep_25.md:13103:textual information based on image pedestrian attributes is effective.
./reviews/deep_25.md:13107:class-level textual features, which are employed as additional alignment cues to facilitate
./reviews/deep_25.md:13110:accuracy. These results validate that contour-guided textual semantics provide meaningful
./reviews/deep_25.md:13114:class-level textual prototypes are independently generated by ATG and CTP, where ATG encodes high-level semantic attributes and CTP captures local structural contours of pedestrians.
./reviews/deep_25.md:13115:These two sets of prototypes are averaged to obtain a unified textual representation, which
./reviews/deep_25.md:13122:configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual
./reviews/deep_25.md:13127:/bin/zsh -lc "pdftotext -f 8 -l 14 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | sed -n '817,1325p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:13561:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:13640:/bin/zsh -lc "pdftotext -f 8 -l 14 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | sed -n '1715,1958p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:13715:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:13845:© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
./reviews/deep_25.md:13894:/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '674,725p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:13896:configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual
./reviews/deep_25.md:13905:/bin/zsh -lc "pdftotext -f 14 -l 15 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '1,120p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_25.md:13957:follows: we utilized the textual information generated from ATG to compute the category
./reviews/deep_25.md:13958:prototypes of the text as alignment cues, guiding images from different modalities to align
./reviews/deep_25.md:13959:towards these cues. As shown in the table, the textual information generated based on image
./reviews/deep_25.md:13963:textual information based on image pedestrian attributes is effective.
./reviews/deep_25.md:13967:class-level textual features, which are employed as additional alignment cues to facilitate
./reviews/deep_25.md:13970:accuracy. These results validate that contour-guided textual semantics provide meaningful
./reviews/deep_25.md:13974:class-level textual prototypes are independently generated by ATG and CTP, where ATG encodes high-level semantic attributes and CTP captures local structural contours of pedestrians.
./reviews/deep_25.md:13975:These two sets of prototypes are averaged to obtain a unified textual representation, which
./reviews/deep_25.md:13982:configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual
./reviews/deep_25.md:14037:2. **重定义动作**：他们没有把问题讲成“把 RGB ReID 迁移到事件数据”，而是讲成“事件 ReID 的核心是 sparse semantic denoising and enhancement”。关键词是 sparse event semantics、event noise、frequency-domain denoising、multi-grain semantic interaction、event contextual semantics。这样一来，旧问题从“缺一个更强 ReID 网络”变成“事件流的语义和噪声强耦合，需要同时去噪和增强上下文”。
./reviews/deep_25.md:14073:2. **重定义动作**：作者把问题讲成“跨模态对齐缺少显式、模态无关的 semantic anchor”，而不是“再做一个 CLIP prompt”。关键词是 attribute-aware text、contour-aware text prompt、text re-definition、category-level textual prototypes、modality-agnostic semantic anchors。这个重定义把文本从辅助描述变成“RGB 和 IR 共同对齐的类别原型”。
./reviews/deep_25.md:14075:3. **机制怎么长出来**：ATG 用 BLIP 问答和固定模板生成属性文本，并刻意避开颜色，保留 age、gender、action、clothing type、hair、glasses 这些更跨模态的属性。CTP 用 SCHP 人体解析去掉衣服区域，保留 body shape 和 pose 相关区域，再用可学习 prompt 表示 contour-aware class text。TRM 负责把 instance-level attribute text 和 class-level contour prompt 融成 image-aware redefined text，最后做 class-level text prototype，指导 RGB 和 IR visual feature 对齐。机制和“属性语义加结构语义”的问题定义是匹配的，但整体偏组合式。
./reviews/deep_25.md:14077:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./reviews/deep_25.md:14079:5. **reviewer 为什么买账**：这篇卖的主要是视角，不是模块精巧度。reviewer 买账点是它把“文本提示”拆成两类可解释语义：属性和轮廓，并用 prototype 把文本变成跨模态锚点。novelty 的来源是“显式语义原型替代弱 token proxy”，而不是 BLIP、SCHP、CLIP 或 Transformer decoder 本身。
./reviews/deep_25.md:14083:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./reviews/deep_25.md:14085:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./reviews/deep_25.md:14087:3. **机制怎么长出来**：SRDA 很直接：先用 MLLM 判断性别，在同一性别内随机选两个 ID，把一个人的上半身和另一个人的下半身横向拼接，生成新 ID，再用微调 BLIP 为新图生成文本。它不追求自然，而追求结构组合多样性。CSRM 则在测试时为文本 query 找 cross-modal k-reciprocal 图像邻居，用 Jaccard 权重把可靠图像特征融合进文本特征，再检索。训练阶段用 CLIP 双流、ID loss、SDM loss 和 image triplet；不加 text triplet，因为 MLLM 文本有噪声。
./reviews/deep_25.md:14089:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reviews/deep_25.md:14099:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reviews/deep_25.md:14103:5. **reviewer 为什么买账**：这篇卖的是“对遮挡失败类型的细分”加“机制一一对应”。如果只说 pose+texture，novelty 会弱；但作者先指出 pose 方法在极端遮挡和行人遮挡下分别为什么失效，再让 PPM 和 TAD 分别处理，reviewer 就更容易接受。真正 novelty 来自问题拆解，而不是 Transformer、pose 或 attention 本身。
./reviews/deep_25.md:14116:2. **重定义动作**：他们没有把问题讲成“把 RGB ReID 迁移到事件数据”，而是讲成“事件 ReID 的核心是 sparse semantic denoising and enhancement”。关键词是 sparse event semantics、event noise、frequency-domain denoising、multi-grain semantic interaction、event contextual semantics。这样一来，旧问题从“缺一个更强 ReID 网络”变成“事件流的语义和噪声强耦合，需要同时去噪和增强上下文”。
./reviews/deep_25.md:14152:2. **重定义动作**：作者把问题讲成“跨模态对齐缺少显式、模态无关的 semantic anchor”，而不是“再做一个 CLIP prompt”。关键词是 attribute-aware text、contour-aware text prompt、text re-definition、category-level textual prototypes、modality-agnostic semantic anchors。这个重定义把文本从辅助描述变成“RGB 和 IR 共同对齐的类别原型”。
./reviews/deep_25.md:14154:3. **机制怎么长出来**：ATG 用 BLIP 问答和固定模板生成属性文本，并刻意避开颜色，保留 age、gender、action、clothing type、hair、glasses 这些更跨模态的属性。CTP 用 SCHP 人体解析去掉衣服区域，保留 body shape 和 pose 相关区域，再用可学习 prompt 表示 contour-aware class text。TRM 负责把 instance-level attribute text 和 class-level contour prompt 融成 image-aware redefined text，最后做 class-level text prototype，指导 RGB 和 IR visual feature 对齐。机制和“属性语义加结构语义”的问题定义是匹配的，但整体偏组合式。
./reviews/deep_25.md:14156:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./reviews/deep_25.md:14158:5. **reviewer 为什么买账**：这篇卖的主要是视角，不是模块精巧度。reviewer 买账点是它把“文本提示”拆成两类可解释语义：属性和轮廓，并用 prototype 把文本变成跨模态锚点。novelty 的来源是“显式语义原型替代弱 token proxy”，而不是 BLIP、SCHP、CLIP 或 Transformer decoder 本身。
./reviews/deep_25.md:14162:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./reviews/deep_25.md:14164:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./reviews/deep_25.md:14166:3. **机制怎么长出来**：SRDA 很直接：先用 MLLM 判断性别，在同一性别内随机选两个 ID，把一个人的上半身和另一个人的下半身横向拼接，生成新 ID，再用微调 BLIP 为新图生成文本。它不追求自然，而追求结构组合多样性。CSRM 则在测试时为文本 query 找 cross-modal k-reciprocal 图像邻居，用 Jaccard 权重把可靠图像特征融合进文本特征，再检索。训练阶段用 CLIP 双流、ID loss、SDM loss 和 image triplet；不加 text triplet，因为 MLLM 文本有噪声。
./reviews/deep_25.md:14168:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reviews/deep_25.md:14178:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reviews/deep_25.md:14182:5. **reviewer 为什么买账**：这篇卖的是“对遮挡失败类型的细分”加“机制一一对应”。如果只说 pose+texture，novelty 会弱；但作者先指出 pose 方法在极端遮挡和行人遮挡下分别为什么失效，再让 PPM 和 TAD 分别处理，reviewer 就更容易接受。真正 novelty 来自问题拆解，而不是 Transformer、pose 或 attention 本身。
./debate/d_2.md:14:一个做行人 ReID 的团队要发 CCF-B 方法稿。连续两个方向被零训练 kill-switch 干净杀死后(航拍-不确定性包含/SMPL-正样本可靠性, 都是错的隐藏变量), 第三个方向 **Gallery Hubness** 的零训练 kill-switch 出了 nuanced 结果, 现在要决定**该不该投入第一次训练**去做训练版。
./debate/d_2.md:18:**零训练 kill-switch 结果(冻结 Market 强 ckpt exp260b, sanity cosine mAP94.61≈训练94.4)**:
./debate/d_2.md:19:- 诊断关全过: ★D4 决定性——负向 in-degree 与'热门样本'**正交**(rho(AP误差,M_neg)=+0.28 但 M_all=−0.08 符号翻转, Spearman(H_neg,H_all)=−0.029 全量近无关)。hub mass 解释 AP 误差完胜 norm/margin/camera/#pos(partial 控住全部仍+0.33)。top1% hub 吃 22-53% false-top1。D1 置换破(增益消失)。novelty-check: ReID 里无确切先例, 但跨模态检索 HAL(CVPR20)/NeighborRetr(CVPR25) 已做训练端 hubness-aware loss+memory bank(任务是图文非 person ReID)。
./debate/d_2.md:25:角色=**蓝队(降级 Hubness)**。为'别投训练, 降级转 r_2 备胎 Rank-Instability'辩护: D2 是铁证——k-reciprocal/camera **免费 test-time** 就在 mAP 上赢, 一个训练版要 beat camera-aware k-reciprocal 是高 bar 且小 headroom(Market 饱和)。用联网查: (a)hubness-aware training / anti-hub margin 历史上是否真能 beat 强 re-ranking(还是总被 k-reciprocal 这类盖过); (b)R1 赢 mAP 输是不是只是把 ranking 重排没真增加判别信息; (c)团队反复投不确定训练的教训。论证'诊断真≠方法能发', 该把诊断当一个 observation 写进别的稿/换 r_2。信心 1-10。
./debate/d_2.md:27:我会按蓝队立场先查一轮原始论文/官方页面，重点看 anti-hub/hubness-aware training 的证据强度，以及 ReID 里 k-reciprocal/camera 这类 test-time 方法的相对门槛。
./debate/d_2.md:29:web search: hubness-aware loss retrieval HAL CVPR 2020 anti-hub margin memory bank paper
./debate/d_2.md:31:web search: "HAL" "hubness" "CVPR" 2020 retrieval
./debate/d_2.md:33:web search: "hubness-aware" "retrieval" "loss"
./debate/d_2.md:71:web search: "hubness" "person re-identification"
./debate/d_2.md:77:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./debate/d_2.md:79:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./debate/d_2.md:83:1. **hubness-aware training 不是空白，但证据不迁移到 ReID。**  
./debate/d_2.md:84:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./debate/d_2.md:92:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./debate/d_2.md:111:- 只有在额外零训练 kill-switch 上同时满足这些门槛才重新升级：OccludedDuke 或 MSMT 上 mAP 增益显著高于 k-reciprocal/camera；hub penalty 与 k-reciprocal 组合后仍有独立 mAP 增益；不是只涨 R1。
./debate/d_2.md:114:信心：**8/10**。不是说 hubness 是错的，而是说“诊断真”到“训练版可发”中间缺的正是 D2 没过的那一关。
./debate/d_2.md:117:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./debate/d_2.md:119:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./debate/d_2.md:123:1. **hubness-aware training 不是空白，但证据不迁移到 ReID。**  
./debate/d_2.md:124:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./debate/d_2.md:132:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./debate/d_2.md:151:- 只有在额外零训练 kill-switch 上同时满足这些门槛才重新升级：OccludedDuke 或 MSMT 上 mAP 增益显著高于 k-reciprocal/camera；hub penalty 与 k-reciprocal 组合后仍有独立 mAP 增益；不是只涨 R1。
./debate/d_2.md:154:信心：**8/10**。不是说 hubness 是错的，而是说“诊断真”到“训练版可发”中间缺的正是 D2 没过的那一关。
./reviews/deep_19.md:14:你是 ReID 创新方法论拆解员 19。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_19.md:52:/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:59:# PDF Skill
./reviews/deep_19.md:73:3. Use `pdfplumber` or `pypdf` for text extraction and quick checks; do not rely on text extraction for layout fidelity.
./reviews/deep_19.md:128:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reviews/deep_19.md:147:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_19.md:148:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_19.md:158:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_19.md:165:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_19.md:178:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_19.md:182:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_19.md:226:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_19.md:233:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_19.md:276:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_19.md:289:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_19.md:346:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_19.md:362:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_19.md:386:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_19.md:396:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_19.md:398:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_19.md:414:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_19.md:420:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_19.md:431:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_19.md:449:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_19.md:475:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_19.md:627:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_19.md:648:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_19.md:664:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_19.md:716:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_19.md:729:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_19.md:756:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_19.md:763:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_19.md:818:我已经读到当前目标还是文献调研和候选方向生成，所以这次拆论文是直接服务于“怎么把观察变成可投稿方法”。下面开始逐篇用 `pdftotext -f 1 -l 8` 抽取前 8 页方法正文，之后如果某篇方法没覆盖完整，我会再补读后续页。
./reviews/deep_19.md:832:/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi Queue for Unsupervised Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:834:/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:872:appearance-based features—such as clothing color distributions, texture patterns, and stylistic attributes—operating under the assumption that subjects retain consistent attire within spatially or temporally constrained environments. While effective in controlled scenarios, these approaches face critical limitations in practical deployments: pedestrians frequently alter clothing across extended time
./reviews/deep_19.md:873:spans or diverse geographical contexts, rendering appearance-centric
./reviews/deep_19.md:889:[19] uses textual descriptions and pre-trained models like CLIP [26]
./reviews/deep_19.md:891:text and image introduces noise.
./reviews/deep_19.md:893:making them valuable for CC-ReID. However, using CLIP introduces a semantic gap between text and image, reducing recognition
./reviews/deep_19.md:908:captions into a text encoder to extract textual features, then apply an
./reviews/deep_19.md:955:enabling efﬁcient image-to-text conversion and generation. QwenVL [1] excels in object detection and localization, providing robust cross-language visual understanding. InternVL2.5 [4] optimizes
./reviews/deep_19.md:974:depth sensing) and risk discarding discriminative color/texture features crucial for short-term identiﬁcation. To mitigate these limitations, hybrid frameworks like CCAF [19] integrate CLIP’s visionlanguage capabilities [26] with textual prompts to guide clothesagnostic feature learning. Nevertheless, the inherent semantic gap
./reviews/deep_19.md:975:between coarse textual descriptions and ﬁne-grained visual patterns
./reviews/deep_19.md:995:Previous studies attempt to extract clothes-agnostic features by introducing text descriptions, focusing on inherent characteristics like
./reviews/deep_19.md:996:body shape while avoiding clothing details. However, these descriptions often lack speciﬁcity and unique identifying features. To address this, we leverage LVLMs to generate detailed clothing descriptions from images. We then use a text encoder to extract clothing features from these descriptions and apply an attention mechanism to reduce the inﬂuence of clothing characteristics on image features. This
./reviews/deep_19.md:1040:input the descriptions and their corresponding images into the pretrained CLIP model’s text and image encoders, generating text features Ft and image features Fv . We calculate the cosine similarity to
./reviews/deep_19.md:1076:processed through CLIP’s text encoder to generate textual features.
./reviews/deep_19.md:1077:The text features from the two descriptions are averaged to produce
./reviews/deep_19.md:1081:the ﬁnal text feature Ft = {fsos
./reviews/deep_19.md:1087:and the text features Ft , resulting in the normalized features Fv and
./reviews/deep_19.md:1514:CLIP text encoder to generate textual embeddings for clothing attributes. To address viewpoint variations, we integrate camera metadata as camera-aware positional encoding following prior work [24],
./reviews/deep_19.md:1835:textures and environmental artifacts, which are inherently unstable
./reviews/deep_19.md:1881:and J. Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint
./reviews/deep_19.md:1884:Zheng. Learning 3d shape feature for texture-insensitive person reidentiﬁcation. In Proceedings of the IEEE/CVF conference on computer
./reviews/deep_19.md:2844:/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-feature balanced network for clothes-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:2892:features. Finally, a cross fusion module (CFM) is used to integrate more contextual information and mine
./reviews/deep_19.md:2934:0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_19.md:2959:can reduce the differences in clothing color and texture information in
./reviews/deep_19.md:2970:color and texture that showed significant performance degradation in
./reviews/deep_19.md:3008:clues. To integrate more contextual information, MBNet introduces a
./reviews/deep_19.md:3076:which makes the model less dependent on clothing colors and textures.
./reviews/deep_19.md:3138:learned features, aiming to dig out more contextual information.
./reviews/deep_19.md:4026:4. In this context, ‘‘C → M’’ refers to the single attention interaction
./reviews/deep_19.md:4064:/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:4078:through attribute recognition to reduce annotation dependency. Using these pseudo-labels and attribute texts, it enhances identity-related attributes while suppressing clothingrelated ones. This is achieved through feature disentanglement at both the caption and attribute levels. Experiments
./reviews/deep_19.md:4103:additional cues like sketches, face, gait, clothing-erased images, attributes, and textual prompts. However, they fail to
./reviews/deep_19.md:4105:texts. AD ViT [14] utilizes only clothing-related attributes,
./reviews/deep_19.md:4108:texts containing richer semantic information. CCAF [16] and
./reviews/deep_19.md:4109:MIPL [17] generate text from images but lack attribute information. Moreover, their reliance on clothing labels increases
./reviews/deep_19.md:4113:labels and attribute texts to compel the model to focus on
./reviews/deep_19.md:4118:guidance in pixel-level disentanglement, an attribute textual disentanglement module is implemented by reinforcing
./reviews/deep_19.md:4291:attribute texts are generated. In the feature extraction module, features are extracted separately from the raw images,
./reviews/deep_19.md:4292:masked images, caption-level texts, and attribute-level texts.
./reviews/deep_19.md:4326:Attribute Text Generation. The attribute labels are converted into attribute texts using hand-crafted templates. For
./reviews/deep_19.md:4327:each image, 5 invariant attribute-level texts Tap are generated (e.g., “A pedestrian under the age of 30”), along with
./reviews/deep_19.md:4328:30 clothing-related attribute-level texts Tan (e.g., “A pedestrian without leather shoes”). Each image is also associated
./reviews/deep_19.md:4329:with one positive caption-level text Tsp that includes age and
./reviews/deep_19.md:4331:caption-level text Tsn that describes other clothing attributes
./reviews/deep_19.md:4376:text encoder to extract textual features at both the caption
./reviews/deep_19.md:4443:image features with the identity-relevant attribute texts. The
./reviews/deep_19.md:4629:Rank-1 by 2.0%, validating multi-granularity textual disentanglement. Masked Branch Disentanglement (M LB).
./reviews/deep_19.md:4754:/bin/zsh -lc "pdftotext -f 4 -l 6 'Multi-feature balanced network for clothes-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:4773:learned features, aiming to dig out more contextual information.
./reviews/deep_19.md:5151:/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:5216:on appearance features, such as the color and texture of
./reviews/deep_19.md:5458:first aggregate features from its neighbors to obtain temporal contextual information:
./reviews/deep_19.md:5526:This update preserves the current part representation while incorporating cross-frame context, strengthening part-wise temporal modeling and robustness to pose
./reviews/deep_19.md:5811:/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:6440:/bin/zsh -lc "pdftotext -f 8 -l 13 'Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:7469:and Huchuan Lu. Tf-clip: Learning text-free clip for video-based
./reviews/deep_19.md:7530:/bin/zsh -lc "pdftotext -f 8 -l 11 'Multi-feature balanced network for clothes-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:7677:4. In this context, ‘‘C → M’’ refers to the single attention interaction
./reviews/deep_19.md:7844:shape feature for texture-insensitive person re-identification. In Proceedings of the
./reviews/deep_19.md:7947:/bin/zsh -lc "pdftotext -f 8 -l 11 'Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:8419:/bin/zsh -lc "pdftotext -f 3 -l 6 'Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_19.md:8466:result. We apply a softmax function to the features in the obtained TopK list and compute a weighted sum to encourage the model’s contextual
./reviews/deep_19.md:8951:1. 触发观察：已有换衣 ReID 往往回避衣服信息，或者用非衣服描述如体型、姿态、轮廓来学习不变特征。但这些非衣服描述经常不精确，CLIP 文本和细粒度图像之间又有语义差距。作者反过来观察到，衣服描述虽然不该成为最终身份特征，但它可以告诉模型“哪些东西应该被压掉”。
./reviews/deep_19.md:8953:2. 重定义动作：旧问题是“如何避免模型依赖衣服”；新问题是“如何准确定位衣服语义并把它从视觉特征中减掉”。这一步很关键，因为作者不是说衣服文本有用来识别人，而是说衣服文本有用来抑制衣服。关键词是 clothing descriptions、coarse-to-fine、semantic-guided、clothes-agnostic、parsing image。
./reviews/deep_19.md:8955:3. 机制怎么长出来：机制和重定义绑定较紧。先用 human parsing 抠出衣服区域，让多个 LVLM 只描述衣服，并用粗粒度和细粒度提示生成多种衣服 caption。再用 CLIP 相似度过滤，保留最可信的两个描述。SGE 分支让图像特征和衣服文本做注意力，然后从视觉特征里减去衣服相关成分。PIE 分支直接遮掉衣服区域训练，让模型被迫使用更稳定的身体、姿态、局部线索。
./reviews/deep_19.md:8975:1. 触发观察：作者看到现有多模态换衣 ReID 没把属性用充分。AD-ViT、MADE 一类用属性标签，但多是一维 one-hot；CCAF、MIPL 一类用文本，但缺少结构化属性信息；很多方法还依赖人工衣服标签。另一个观察是，像素级遮衣服能做低层解耦，但缺少高层语义指导。
./reviews/deep_19.md:8977:2. 重定义动作：旧问题是“属性能不能作为辅助标签”；新问题是“属性文本能不能作为多粒度提示，明确规定哪些语义该保留、哪些该压制”。关键词是 multi-granularity attribute prompt、attribute-level、caption-level、identity-related、clothing-related、pseudo label。
./reviews/deep_19.md:8979:3. 机制怎么长出来：机制很顺。先用 SCHP 生成衣服擦除图，再用 AAPAR 识别 35 类属性。年龄、性别被当作身份相关属性，头饰、上衣、下装、鞋、配饰等被当作衣服相关属性。然后用模板把属性变成文本，在 raw image branch 和 masked image branch 上用 CLIP ViT 加 visual deep prompt 学图像表示。属性解耦部分把图像特征和身份相关文本拼接，用 triplet 促进保留；对衣服相关文本用负向损失或对抗约束进行压制。BNNeck 用来隔开图像调优和文本对齐，避免一个目标破坏另一个目标。
./reviews/deep_19.md:8981:4. 证据闭环：消融很对题。baseline 加 clothing adversarial attribute 后提升，说明压制衣服属性有效；加 identity-relevant attribute 后继续提升，说明保留稳定属性有效；再加 caption-level 约束提升，说明多粒度文本不是摆设；masked branch 的属性解耦去掉后也下降，说明擦衣服后仍有残留衣服语义。检索可视化显示 baseline 受颜色和款式影响更重，MAPL 更能跨衣服找人。
./reviews/deep_19.md:8983:5. 审稿人为什么买账：这篇卖的是视角。它不是“又用了属性”，而是把属性拆成正向身份语义和负向衣服语义，再用文本提示做多粒度解耦。真正新意来自“属性从标签变成 prompt 化的解耦约束”，同时减少人工衣服标签依赖。
./reviews/deep_19.md:9028:1. 触发观察：已有换衣 ReID 往往回避衣服信息，或者用非衣服描述如体型、姿态、轮廓来学习不变特征。但这些非衣服描述经常不精确，CLIP 文本和细粒度图像之间又有语义差距。作者反过来观察到，衣服描述虽然不该成为最终身份特征，但它可以告诉模型“哪些东西应该被压掉”。
./reviews/deep_19.md:9030:2. 重定义动作：旧问题是“如何避免模型依赖衣服”；新问题是“如何准确定位衣服语义并把它从视觉特征中减掉”。这一步很关键，因为作者不是说衣服文本有用来识别人，而是说衣服文本有用来抑制衣服。关键词是 clothing descriptions、coarse-to-fine、semantic-guided、clothes-agnostic、parsing image。
./reviews/deep_19.md:9032:3. 机制怎么长出来：机制和重定义绑定较紧。先用 human parsing 抠出衣服区域，让多个 LVLM 只描述衣服，并用粗粒度和细粒度提示生成多种衣服 caption。再用 CLIP 相似度过滤，保留最可信的两个描述。SGE 分支让图像特征和衣服文本做注意力，然后从视觉特征里减去衣服相关成分。PIE 分支直接遮掉衣服区域训练，让模型被迫使用更稳定的身体、姿态、局部线索。
./reviews/deep_19.md:9052:1. 触发观察：作者看到现有多模态换衣 ReID 没把属性用充分。AD-ViT、MADE 一类用属性标签，但多是一维 one-hot；CCAF、MIPL 一类用文本，但缺少结构化属性信息；很多方法还依赖人工衣服标签。另一个观察是，像素级遮衣服能做低层解耦，但缺少高层语义指导。
./reviews/deep_19.md:9054:2. 重定义动作：旧问题是“属性能不能作为辅助标签”；新问题是“属性文本能不能作为多粒度提示，明确规定哪些语义该保留、哪些该压制”。关键词是 multi-granularity attribute prompt、attribute-level、caption-level、identity-related、clothing-related、pseudo label。
./reviews/deep_19.md:9056:3. 机制怎么长出来：机制很顺。先用 SCHP 生成衣服擦除图，再用 AAPAR 识别 35 类属性。年龄、性别被当作身份相关属性，头饰、上衣、下装、鞋、配饰等被当作衣服相关属性。然后用模板把属性变成文本，在 raw image branch 和 masked image branch 上用 CLIP ViT 加 visual deep prompt 学图像表示。属性解耦部分把图像特征和身份相关文本拼接，用 triplet 促进保留；对衣服相关文本用负向损失或对抗约束进行压制。BNNeck 用来隔开图像调优和文本对齐，避免一个目标破坏另一个目标。
./reviews/deep_19.md:9058:4. 证据闭环：消融很对题。baseline 加 clothing adversarial attribute 后提升，说明压制衣服属性有效；加 identity-relevant attribute 后继续提升，说明保留稳定属性有效；再加 caption-level 约束提升，说明多粒度文本不是摆设；masked branch 的属性解耦去掉后也下降，说明擦衣服后仍有残留衣服语义。检索可视化显示 baseline 受颜色和款式影响更重，MAPL 更能跨衣服找人。
./reviews/deep_19.md:9060:5. 审稿人为什么买账：这篇卖的是视角。它不是“又用了属性”，而是把属性拆成正向身份语义和负向衣服语义，再用文本提示做多粒度解耦。真正新意来自“属性从标签变成 prompt 化的解耦约束”，同时减少人工衣服标签依赖。
./reviews/deep_14.md:14:你是 ReID 创新方法论拆解员 14。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_14.md:31:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_14.md:36:我先按项目规则确认目标和论文文件，然后用 `pdftotext` 逐篇抽前 8 页的方法部分。这个任务不需要改文件，我会只读本地 PDF 和必要的公开背景信息。
./reviews/deep_14.md:91:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_14.md:118:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_14.md:147:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_14.md:162:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_14.md:172:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_14.md:181:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_14.md:187:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_14.md:202:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_14.md:212:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_14.md:214:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_14.md:237:-rw-r--r--@   1 abdslm  staff   3807986 Jun  9 21:33 A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_14.md:238:-rw-r--r--@   1 abdslm  staff   3976252 Jun  9 20:34 A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_14.md:248:-rw-r--r--@   1 abdslm  staff   3772906 Jun  9 22:26 Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_14.md:255:-rw-r--r--@   1 abdslm  staff   2749258 Jun  9 22:42 CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_14.md:268:-rw-r--r--@   1 abdslm  staff   9515723 Jun 10 00:22 ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_14.md:272:-rw-r--r--@   1 abdslm  staff   2147219 Jun  9 16:58 Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_14.md:316:-rw-r--r--@   1 abdslm  staff   6934377 Jun  9 23:35 Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_14.md:323:-rw-r--r--@   1 abdslm  staff   4656109 Jun  9 17:03 Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_14.md:366:-rw-r--r--@   1 abdslm  staff   3667918 Jun  9 04:09 See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_14.md:379:-rw-r--r--@   1 abdslm  staff   4657084 Jun  9 18:38 TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_14.md:471:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_14.md:484:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_14.md:511:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_14.md:518:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_14.md:673:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_14.md:675:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_14.md:696:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_14.md:712:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_14.md:770:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_14.md:771:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_14.md:795:/bin/zsh -lc "pdftotext -f 1 -l 8 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|conclusion|geospatial|temporal|graph'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:802:35:can integrate geospatial constraint and spatio-temporal contextual
./reviews/deep_14.md:806:57:can integrate geospatial constraint and spatio-temporal contextual information. By modeling the appearance of groups across
./reviews/deep_14.md:872:544:spanning group-level context modeling (MACG [5]), classical graph
./reviews/deep_14.md:889:599:“Group re-identification with group context graph neural networks,” IEEE Transactions on Multimedia, vol. 23, pp. 2614–
./reviews/deep_14.md:890:602:“Learning multi-attention context graph for group-based reidentification,” IEEE Transactions on Pattern Analysis and
./reviews/deep_14.md:902:/bin/zsh -lc "pdftotext -f 1 -l 8 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|vision|language|vlm|pseudo|knowledge'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:927:103:textual semantics, facilitating the integration of semanticlevel understanding into pseudo-label generation; on the other
./reviews/deep_14.md:938:153:each instance can be assigned as a pseudo-ID, whose text
./reviews/deep_14.md:977:322:feature space. Text prompts are generated from predefined sets of attributes, and pseudo-IDs are assigned based on the highest similarity scores between text
./reviews/deep_14.md:979:326:leveraging the CLIP text–image matching principle to generate pseudo-IDs. These pseudo-IDs are subsequently refined
./reviews/deep_14.md:1059:/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|heterogeneous|generative|token|distance|recovery'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:1237:/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|hierarchical|fusion|local-aware|local aware|transformer'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:1284:156:the representation ability of local features by employing a pixel context encoder and a local prototype decoder. Lin et al. (2024) proposed a multi-level relation-aware Transformer (MLRAT), which has
./reviews/deep_14.md:1287:161:with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reviews/deep_14.md:1301:194:proposed an improved YOLOv11 architecture for the automated classiﬁcation of peripheral blood cells, which integrates a Dynamic CrossScale Context Aggregation (DCSCA) module. Through parallel convolution, dynamic attention, and cross-scale interaction, the module enables
./reviews/deep_14.md:1367:/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|hierarchical|knowledge|reasoning|text|image|attribute|semantic'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:1369:10:Hierarchical knowledge-guided reasoning for text-based person
./reviews/deep_14.md:1372:38:Masked language modeling (MLM) has expanded the exploration of text-image person re-identiﬁcation (TIReID)
./reviews/deep_14.md:1373:39:tasks from coarse-granularity to ﬁne-grained alignment. Whereas, we note that vanilla MLM picks random tokens for visual-to-token reasoning, which could fail the intention of semantic visual-textual alignment by indistinguishably focusing on all the sub-words. This work proposes to leverage the inherent hierarchical scene
./reviews/deep_14.md:1374:40:graph knowledge in each text for guiding token masking and enhancing cross-modal representation in TIReID,
./reviews/deep_14.md:1375:41:thus relieving the pitfall of blind visual-textual alignment. The proposed framework, Hierarchical KnowledgeGuided Reasoning (HKGR), parses object-level, attribute-level, and relation-level masking according to phrase
./reviews/deep_14.md:1377:43:which leverages the token selection method and image-text similarity distribution constraint to further facilitate
./reviews/deep_14.md:1378:44:the semantic alignment between image and text at both coarse-grained and ﬁne-grained levels. Experimental
./reviews/deep_14.md:1383:51:searching pedestrian images with the same identity according to text
./reviews/deep_14.md:1384:56:alignment to bridge the inherent modality gap between images and text
./reviews/deep_14.md:1386:60:global-matching methods, separately maps the visual global representation and textual global representation into a joint embedding space
./reviews/deep_14.md:1387:63:Nevertheless, it can be hard for such a compact representation to capture ﬁne-grained semantic details in texts and images. For example, as
./reviews/deep_14.md:1388:64:is shown in Fig. 1, understanding the text descriptions involves complicated semantic reasoning regarding diﬀerent objects (‘lady’, ‘pant’,
./reviews/deep_14.md:1390:67:Ding et al., 2021; Wang et al., 2020b, 2022a; Yan et al., 2023a), localmatching methods, leverages detailed visual cues and individual textual words to represent images and texts, respectively, and performs
./reviews/deep_14.md:1392:70:2023; Jiang & Ye, 2023; Zuo et al., 2023), which fall under the localmatching methods category, adopt local semantic reasoning to establish ﬁne-grained relationships between image and text representations,
./reviews/deep_14.md:1393:79:0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_14.md:1395:86:(e.g., “in front of”), leading to fragmented textual representations and
./reviews/deep_14.md:1396:87:an increased semantic gap between global textual descriptions and local
./reviews/deep_14.md:1399:92:knowledge in scene graphs parsed from text to enhance the masking
./reviews/deep_14.md:1401:94:we decompose text descriptions into three hierarchical semantic levels
./reviews/deep_14.md:1404:99:to mitigate semantic distortion caused by random masking (for Challenge 1). Moreover, for the KGR module, we propose object-level reasoning, attribute-level reasoning, and relation-level reasoning tasks, realizing multi-level cross-modal consistency alignment to narrow the semantic gap between image and text (for Challenge 2). Furthermore, we
./reviews/deep_14.md:1406:101:ﬁrst utilizes a token selection method to select multi-grained discriminative information both in visual and text tokens, and then it constrains
./reviews/deep_14.md:1407:102:the multi-grained image-text similarity distributions for proper crossmodal alignments at both coarse-grained and ﬁne-grained levels. Our
./reviews/deep_14.md:1409:107:framework, which decomposes text descriptions into three levels,
./reviews/deep_14.md:1411:111:the semantic gap between image and text, we design a KnowledgeGuided Reasoning (KGR) module for better semantic coverage.
./reviews/deep_14.md:1414:114:by multi-grained image-text similarity distributions.
./reviews/deep_14.md:1416:121:(a) The MLM-based paradigm is trained by randomly masking and predicting sub-words based on the unmasked contextual texts and the paired image
./reviews/deep_14.md:1419:127:us to investigate MLM in the context of TIReID tasks, aiming to exploit
./reviews/deep_14.md:1420:128:the ﬁne-grained semantic interaction across images and texts.
./reviews/deep_14.md:1421:130:predicting text sub-words to align contextual information with image
./reviews/deep_14.md:1422:132:random masking strategy is often suboptimal for image-text alignment
./reviews/deep_14.md:1423:134:of semantic concepts in textual descriptions. In many TIReID benchmarks (Ding et al., 2021; Li et al., 2017b; Zhu et al., 2021), it is common for multiple textual descriptions to refer to the same individual
./reviews/deep_14.md:1424:136:Therefore, the ﬁne-grained diﬀerences among text descriptions bring
./reviews/deep_14.md:1425:137:challenges for image-text alignment in MLM-based TIReID paradigms:
./reviews/deep_14.md:1426:138:1) The randomness of masked language: Given the complexity of semantic information in textual descriptions, vanilla MLM-based methods
./reviews/deep_14.md:1429:142:Consequently, the learned visual-textual correspondences may become
./reviews/deep_14.md:1436:153:2024; Ye et al., 2021; Zheng et al., 2016) and image-text retrieval (Faghri et al., 2017; Qin et al., 2022; Zheng et al., 2020), was
./reviews/deep_14.md:1438:158:proper alignment between image and text modalities in the common
./reviews/deep_14.md:1439:160:networks to extract image and text features and align these features
./reviews/deep_14.md:1440:161:using an image-text contrastive loss in the shared latent space. Zheng
./reviews/deep_14.md:1441:168:et al. (2020) propose a Dual-Path method that employs a CNN structure for both image and text feature extraction to enable eﬀective endto-end ﬁne-tuning using an instance loss. Zhang and Lu (2018) design
./reviews/deep_14.md:1443:175:a dual-path local alignment network to extract local visual and textual representations from horizontally segmented image patches. Subsequently, local representations are aligned adaptively with a multi-stage
./reviews/deep_14.md:1445:180:the local ﬁne-grained correspondence between image-text modalities.
./reviews/deep_14.md:1446:185:visual-textual tokens. Speciﬁcally, Jiang and Ye (2023) proposes an implicit relation reasoning module to predict the random masked text tokens based on image patches and unmasked surrounding text tokens,
./reviews/deep_14.md:1447:186:aiming to align images and text representation. In the work of FLIP (Zuo
./reviews/deep_14.md:1449:188:masked textual embeddings with global image embeddings to construct
./reviews/deep_14.md:1450:189:correlations between images and texts.
./reviews/deep_14.md:1452:192:in scene graphs from text, aiming to guide detailed semantic alignment
./reviews/deep_14.md:1453:193:across visual and text modalities.
./reviews/deep_14.md:1458:208:extract image and text features separately with distinct encoders. These
./reviews/deep_14.md:1460:212:been applied to various downstream multi-modal tasks, including textvideo retrieval (Fang et al., 2021; Luo et al., 2022; Ma et al., 2022)
./reviews/deep_14.md:1462:215:into video-text cross-modal retrieval and demonstrate that a large-scale
./reviews/deep_14.md:1463:216:multi-modal pre-training model is beneﬁcial for video-text retrieval.
./reviews/deep_14.md:1466:229:follows, we ﬁrst introduce the image-text feature representation in Section 3.1, and then describe the scene graph construction in Section 3.2,
./reviews/deep_14.md:1474:245:introduce both visual scene graphs and textual scene graphs to characterize objects and their relations, optimizing the model by computing
./reviews/deep_14.md:1475:248:that combines a visual scene graph and textual scene graph to enable the
./reviews/deep_14.md:1476:250:3.1. Image-text feature representation
./reviews/deep_14.md:1483:265:Text feature representation. In terms of text feature representation, we adopt the text encoder of pre-trained CLIP as a text feature
./reviews/deep_14.md:1484:272:Fig. 2. Overview of the proposed HKGR: given image-text pairs, the image is encoded into an embedding by the image encoder. Simultaneously, the text is parsed
./reviews/deep_14.md:1485:273:into a scene graph, and then a knowledge-guided masking strategy is adopted to generate masked text. Finally, the masked text embedding obtained from the
./reviews/deep_14.md:1486:274:text encoder and image embedding are fed into a cross-modal interaction encoder to perform object-level reasoning, attribute-level reasoning, and relation-level
./reviews/deep_14.md:1488:289:extractor. Speciﬁcally, for the input text 𝑇𝑖 , we use the lower-cased byte
./reviews/deep_14.md:1489:290:pair encoding (BPE) tokenizer to tokenize 𝑇𝑖 into a sequence of text tokens. We then add the speciﬁc token [SOS] at the start and [EOS] at the
./reviews/deep_14.md:1490:292:transformer blocks of the text encoder to obtain the ﬁnal text features
./reviews/deep_14.md:1491:296:Cross-modal interaction encoder. In order to construct the interaction between image and text modality, we adopt a transformer-based
./reviews/deep_14.md:1492:297:encoder for image and text features fusion. As shown in Fig. 3, the CrossModal Interaction Encoder(CME) contains transformer blocks composed
./reviews/deep_14.md:1493:299:Given a text 𝑇𝑖 , we mask a portion of the keywords/phrases derived
./reviews/deep_14.md:1494:300:from the scene graph parsed from the text to obtain masked text 𝑇̂𝑖 . The
./reviews/deep_14.md:1495:301:paired image 𝐼𝑖 and masked text 𝑇̂𝑖 are then fed into the image/text encoder to output image feature tokens 𝑓𝑖𝑣 and masked text feature tokens
./reviews/deep_14.md:1496:302:𝑓̂𝑖𝑡 = {𝑡̂𝑖𝑠𝑜𝑠 , 𝑡̂𝑖1 , 𝑡̂𝑖2 , …, 𝑡̂𝑖𝑛 , 𝑡̂𝑖𝑒𝑜𝑠 }. Image feature tokens 𝑓𝑖𝑣 and masked text fea𝑡
./reviews/deep_14.md:1497:306:denotes multi-head cross-attention, which adopts masked text feature
./reviews/deep_14.md:1501:315:graph. Speciﬁcally, Given an input text 𝑇𝑖 , the scene graph can be represented as 𝐺(𝑇𝑖 ) = {𝑉 (𝑇𝑖 ), 𝐸(𝑇𝑖 )}, where 𝑉 (𝑇𝑖 ) = {𝑂, 𝐴, 𝑅} denote the
./reviews/deep_14.md:1506:320:corresponding node. The scene graph parsed from the text is presented
./reviews/deep_14.md:1509:339:Illustration of scene graph parsed from text.
./reviews/deep_14.md:1510:340:text: 𝑇𝑖
./reviews/deep_14.md:1515:371:given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the relation triplet 𝑄 = {{𝑜1𝑖 , 𝑟𝑖 , 𝑜2𝑖 }|𝑖 = 1, 2, … , 𝑛𝑞 } from the scene graph of text,
./reviews/deep_14.md:1517:377:node pairs {𝑜̂1𝑖 , 𝑜̂2𝑖 } and mask the sub-sequences of the text that correspond to the selected relation nodes {̂𝑟𝑖 }. At last, the relation masked
./reviews/deep_14.md:1518:378:text 𝑇̂𝑖𝑟 and image 𝑇𝑖 are fed into the text/image encoder to generate
./reviews/deep_14.md:1519:379:masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 .
./reviews/deep_14.md:1521:382:masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 . The loss function can be
./reviews/deep_14.md:1522:386:text, we construct a knowledge-guided reasoning module, including object-level reasoning, attribute-level reasoning, and relationlevel reasoning. These reasoning processes guide the model to
./reviews/deep_14.md:1523:387:learn hierarchical semantic local alignments between image and text
./reviews/deep_14.md:1525:390:visual scene and thus play a crucial role in constructing semantic information. Through object-level reasoning, the model is learning to establish the object-level visual-text connection. Given the paired image 𝐼𝑖
./reviews/deep_14.md:1526:391:and text 𝑇𝑖 , we ﬁrst obtain all the object nodes 𝑂 = {𝑜𝑖 |𝑖 = 1, 2, … , 𝑛𝑜 }
./reviews/deep_14.md:1527:392:from the scene graph of text. We randomly select 30 % of object nodes
./reviews/deep_14.md:1528:395:object masking by masking the sub-sequences in the text that correspond to the selected object nodes 𝑂̂ = {𝑜̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑜 }. Finally, the
./reviews/deep_14.md:1529:396:object masked text 𝑇̂𝑖𝑜 and image 𝑇𝑖 are fed into the image-text encoder
./reviews/deep_14.md:1530:397:to obtain masked text features𝑓̂𝑖𝑜,𝑡 and image features 𝑓𝑖𝑣 .
./reviews/deep_14.md:1532:399:object nodes 𝑂̂ tokens based on the masked text features𝑓̂𝑜,𝑡
./reviews/deep_14.md:1536:411:between image and text modalities, thereby hindering retrieval performance gains. To tackle this limitation, we propose a Multi-Grained Semantic Alignment (MGA) module to mine discriminative feature representations and perform multi-grained cross-modal alignments. The MAG
./reviews/deep_14.md:1541:416:from the text encoder as coarse-grained feature representation. Then,
./reviews/deep_14.md:1543:418:KL divergence between global image-text feature similarity distribution
./reviews/deep_14.md:1546:447:Given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the attribute pairs 𝑃 = {{𝑜𝑖 , 𝑎𝑖 }|𝑖 = 1, 2, … , 𝑛𝑝 } from the scene graph of text,
./reviews/deep_14.md:1551:453:masking the sub-sequences of the text that correspond to the selected
./reviews/deep_14.md:1553:455:the attribute masked text 𝑇̂𝑖𝑎 and image 𝑇𝑖 are fed into the text/image
./reviews/deep_14.md:1554:456:encoder to generate masked text features𝑓̂𝑖𝑎,𝑡 and image features 𝑓𝑖𝑣 .
./reviews/deep_14.md:1556:458:nodes {𝑎̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑎 } tokens based on the masked text features𝑓̂𝑖𝑎,𝑡
./reviews/deep_14.md:1560:508:both image and text modalities using correlation weights. In practice,
./reviews/deep_14.md:1562:522:the token selection ratio. In terms of textual procedure, we can obtain
./reviews/deep_14.md:1563:534:the selected local text features in a similar way as 𝑓̂𝑖𝑡 = {𝑡𝑖 𝑡 , 𝑡𝑖 𝑡 , …, 𝑡𝑖 𝑡 },
./reviews/deep_14.md:1564:539:2021) dataset consists of 20,505 pedestrian images and 41,010 text descriptions with two captions per image. The average length of text descriptions is 23 words. The dataset is divided into three subsets, with
./reviews/deep_14.md:1565:545:and textual token features 𝑓̂𝑖𝑡 , we perform linear feature transformation
./reviews/deep_14.md:1566:549:and text embedding 𝑓𝑙𝑜𝑐
./reviews/deep_14.md:1567:579:list, given a text description as the query. In addition, for a more thorough assessment, we also leverage the mean Average Precision (mAP) as
./reviews/deep_14.md:1570:589:text feature similarity distributions 𝜓(𝑆(𝑓𝑙𝑜𝑐
./reviews/deep_14.md:1572:619:We initialized the image and text encoder with pre-trained weights
./reviews/deep_14.md:1574:626:For text data augmentation, we apply random masking, replacement,
./reviews/deep_14.md:1581:639:Following the previous work (Zheng et al., 2020), we also adopt an instance loss 𝑖𝑑 that classiﬁes each image and text into speciﬁc classes
./reviews/deep_14.md:1582:640:based on their corresponding individual IDs, allowing the model to differentiate between image and text groups corresponding to distinct individuals. Above all, the ﬁnal loss function can be formulated as follows:
./reviews/deep_14.md:1585:647:model under texts (e.g., BERT), to encode the image and text respectively: Dual-Path (Zheng et al., 2020), CMPM/C (Zhang & Lu,
./reviews/deep_14.md:1586:656:(e.g., CLIP) to encode images and text: CFine (Yan et al., 2023a), TextReID (Han et al., 2021), VGSG He et al. (2023), IRRA (Jiang & Ye,
./reviews/deep_14.md:1590:668:The CUHK-PEDES (Li et al., 2017b) dataset consists of 40,206 pedestrian images, 80,412 text descriptions and 13,003 individuals, with each
./reviews/deep_14.md:1591:669:image corresponding to two textual captions. The average number of
./reviews/deep_14.md:1592:670:words in each textual description is 23.5. In line with the oﬃcial data
./reviews/deep_14.md:1593:672:person images and 68,108 text descriptions. The validation set includes
./reviews/deep_14.md:1594:673:1000 identities with 3078 person images and 6158 text descriptions,
./reviews/deep_14.md:1596:675:6156 text descriptions. The ICFG-PEDES (Ding et al., 2021) is collected
./reviews/deep_14.md:1597:676:from the MSMT17 (Wei et al., 2018) dataset, which includes 4102 individuals with 54,522 person images and 54,522 text descriptions. The
./reviews/deep_14.md:1599:680:and text descriptions, while the test set consists of 1000 identities with
./reviews/deep_14.md:1600:681:19,848 person images and text descriptions. The RSTPReid (Zhu et al.,
./reviews/deep_14.md:1616:1020:Tarashima, 2023) by 1.46 %, 1.02 %, and 0.48 %, respectively. Compared with the recent strong baseline method IRRA (Jiang & Ye,
./reviews/deep_14.md:1618:1023:• 3) We also conduct our experiment on the newly-published RSTPReid dataset, as depicted in Table 4. Our proposal suppresses
./reviews/deep_14.md:1624:1040:Performance comparisons with SoTA methods on the RSTPReid
./reviews/deep_14.md:1627:1050:achieves 1.83 % improvements of R@1 over the recent vanilla MLMbased model IRRA (Jiang & Ye, 2023). This can be attributed to
./reviews/deep_14.md:1629:1052:thereby reducing the semantic gap between image and text modalities.
./reviews/deep_14.md:1639:1274:attribute-level reasoning, and relation-level reasoning to establish ﬁnegrained connections between images and text. To further evaluate the
./reviews/deep_14.md:1649:1308:text, we use t-SNE to transform the high-dimensional image and text
./reviews/deep_14.md:1658:/bin/zsh -lc "for f in 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf'; do pdfinfo \""'$f" | rg '"'Title|Pages'; done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:1668:Title:           Hierarchical knowledge-guided reasoning for text-based person re-identification
./reviews/deep_14.md:1674:/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|hierarchical|proxy|cloth|clothes|changing|identity|pseudo'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:1873:/bin/zsh -lc "pdftotext -f 1 -l 8 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' - | nl -ba | sed -n '33,351p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:1877:    35	can integrate geospatial constraint and spatio-temporal contextual
./reviews/deep_14.md:1899:    57	can integrate geospatial constraint and spatio-temporal contextual information. By modeling the appearance of groups across
./reviews/deep_14.md:2196:/bin/zsh -lc "pdftotext -f 1 -l 8 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | nl -ba | sed -n '70,599p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:2231:   103	textual semantics, facilitating the integration of semanticlevel understanding into pseudo-label generation; on the other
./reviews/deep_14.md:2238:   110	2162-237X © 2026 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
./reviews/deep_14.md:2248:   120	and a CLIP ID labeler that aligns text and image features
./reviews/deep_14.md:2279:   151	and text features. Specifically, we first construct one sentence
./reviews/deep_14.md:2281:   153	each instance can be assigned as a pseudo-ID, whose text
./reviews/deep_14.md:2323:   195	exploit contextual information to derive discriminative features
./reviews/deep_14.md:2370:   242	which utilizes learnable text tokens for identity description,
./reviews/deep_14.md:2372:   244	image–text matching task. Human-centric models like UniHCP, PATH, and SOLIDER demonstrate the potential of
./reviews/deep_14.md:2450:   322	feature space. Text prompts are generated from predefined sets of attributes, and pseudo-IDs are assigned based on the highest similarity scores between text
./reviews/deep_14.md:2453:   325	a frozen CLIP text encoder and a trainable CLIP projector,
./reviews/deep_14.md:2454:   326	leveraging the CLIP text–image matching principle to generate pseudo-IDs. These pseudo-IDs are subsequently refined
./reviews/deep_14.md:2473:   345	investigation into the customization of text prompts. Typically,
./reviews/deep_14.md:2479:   351	The text prompt is specifically crafted as follows: “This is a
./reviews/deep_14.md:2483:   355	text prompts are subsequently processed by the CLIP text
./reviews/deep_14.md:2484:   356	encoder to generate the text embedding D = [D1 , D2 , . . . , D M ],
./reviews/deep_14.md:2485:   357	where D ∈ R M × C and M signifies the total number of text
./reviews/deep_14.md:2491:   363	I b ∈ RN × C derived from the CLIP projector, and text embeddings D ∈ R M × C obtained from the CLIP text encoder, where
./reviews/deep_14.md:2714:   586	the text embeddings D0 by constructing the text prompt
./reviews/deep_14.md:2717:   589	the similarity between Iib and the text embeddings D0 .
./reviews/deep_14.md:2730:/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '90,920p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:2746:   104	external tools. In this context, ViT [19] has attracted attention
./reviews/deep_14.md:2748:   106	ViT can capture global and local context information through
./reviews/deep_14.md:2753:   111	1051-8215 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
./reviews/deep_14.md:2888:   246	(RGA) module to capture global context information for better
./reviews/deep_14.md:3013:   371	stream, we employ the ViT-B/16 [19] to capture contextual
./reviews/deep_14.md:3062:   420	∈ R(N +1)×D , which contains rich global context information
./reviews/deep_14.md:3167:   525	key to distinguish different pedestrians is to capture discriminative local details. These details, such as the texture, color
./reviews/deep_14.md:3183:   541	enable our model to capture richer local context information,
./reviews/deep_14.md:3262:   620	to inherit the global context information. Then, we project the
./reviews/deep_14.md:3292:   650	global context semantics.
./reviews/deep_14.md:3575:/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '356,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:3592:   371	stream, we employ the ViT-B/16 [19] to capture contextual
./reviews/deep_14.md:3641:   420	∈ R(N +1)×D , which contains rich global context information
./reviews/deep_14.md:3746:   525	key to distinguish different pedestrians is to capture discriminative local details. These details, such as the texture, color
./reviews/deep_14.md:3762:   541	enable our model to capture richer local context information,
./reviews/deep_14.md:3788:/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '899,990p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:3860:/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '49,552p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:3893:    80	0893-6080/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
./reviews/deep_14.md:3905:    92	of them possess a strong ability to capture global contextual information, their capacity for acquiring ﬁne-grained local features is largely
./reviews/deep_14.md:3969:   156	the representation ability of local features by employing a pixel context encoder and a local prototype decoder. Lin et al. (2024) proposed a multi-level relation-aware Transformer (MLRAT), which has
./reviews/deep_14.md:3974:   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reviews/deep_14.md:4007:   194	proposed an improved YOLOv11 architecture for the automated classiﬁcation of peripheral blood cells, which integrates a Dynamic CrossScale Context Aggregation (DCSCA) module. Through parallel convolution, dynamic attention, and cross-scale interaction, the module enables
./reviews/deep_14.md:4008:   195	multi-scale feature capture, scale recalibration, and context modeling,
./reviews/deep_14.md:4283:   470	garment textures and accessory details) manifests predominantly within
./reviews/deep_14.md:4284:   471	speciﬁc semantic regions (e.g., the textural correlation between cuﬀs
./reviews/deep_14.md:4380:/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | nl -ba | sed -n '250,664p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:4382:   250	3.1. Image-text feature representation
./reviews/deep_14.md:4397:   265	Text feature representation. In terms of text feature representation, we adopt the text encoder of pre-trained CLIP as a text feature
./reviews/deep_14.md:4404:   272	Fig. 2. Overview of the proposed HKGR: given image-text pairs, the image is encoded into an embedding by the image encoder. Simultaneously, the text is parsed
./reviews/deep_14.md:4405:   273	into a scene graph, and then a knowledge-guided masking strategy is adopted to generate masked text. Finally, the masked text embedding obtained from the
./reviews/deep_14.md:4406:   274	text encoder and image embedding are fed into a cross-modal interaction encoder to perform object-level reasoning, attribute-level reasoning, and relation-level
./reviews/deep_14.md:4421:   289	extractor. Speciﬁcally, for the input text 𝑇𝑖 , we use the lower-cased byte
./reviews/deep_14.md:4422:   290	pair encoding (BPE) tokenizer to tokenize 𝑇𝑖 into a sequence of text tokens. We then add the speciﬁc token [SOS] at the start and [EOS] at the
./reviews/deep_14.md:4424:   292	transformer blocks of the text encoder to obtain the ﬁnal text features
./reviews/deep_14.md:4428:   296	Cross-modal interaction encoder. In order to construct the interaction between image and text modality, we adopt a transformer-based
./reviews/deep_14.md:4429:   297	encoder for image and text features fusion. As shown in Fig. 3, the CrossModal Interaction Encoder(CME) contains transformer blocks composed
./reviews/deep_14.md:4431:   299	Given a text 𝑇𝑖 , we mask a portion of the keywords/phrases derived
./reviews/deep_14.md:4432:   300	from the scene graph parsed from the text to obtain masked text 𝑇̂𝑖 . The
./reviews/deep_14.md:4433:   301	paired image 𝐼𝑖 and masked text 𝑇̂𝑖 are then fed into the image/text encoder to output image feature tokens 𝑓𝑖𝑣 and masked text feature tokens
./reviews/deep_14.md:4434:   302	𝑓̂𝑖𝑡 = {𝑡̂𝑖𝑠𝑜𝑠 , 𝑡̂𝑖1 , 𝑡̂𝑖2 , …, 𝑡̂𝑖𝑛 , 𝑡̂𝑖𝑒𝑜𝑠 }. Image feature tokens 𝑓𝑖𝑣 and masked text fea𝑡
./reviews/deep_14.md:4438:   306	denotes multi-head cross-attention, which adopts masked text feature
./reviews/deep_14.md:4447:   315	graph. Speciﬁcally, Given an input text 𝑇𝑖 , the scene graph can be represented as 𝐺(𝑇𝑖 ) = {𝑉 (𝑇𝑖 ), 𝐸(𝑇𝑖 )}, where 𝑉 (𝑇𝑖 ) = {𝑂, 𝐴, 𝑅} denote the
./reviews/deep_14.md:4452:   320	corresponding node. The scene graph parsed from the text is presented
./reviews/deep_14.md:4471:   339	Illustration of scene graph parsed from text.
./reviews/deep_14.md:4472:   340	text: 𝑇𝑖
./reviews/deep_14.md:4503:   371	given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the relation triplet 𝑄 = {{𝑜1𝑖 , 𝑟𝑖 , 𝑜2𝑖 }|𝑖 = 1, 2, … , 𝑛𝑞 } from the scene graph of text,
./reviews/deep_14.md:4509:   377	node pairs {𝑜̂1𝑖 , 𝑜̂2𝑖 } and mask the sub-sequences of the text that correspond to the selected relation nodes {̂𝑟𝑖 }. At last, the relation masked
./reviews/deep_14.md:4510:   378	text 𝑇̂𝑖𝑟 and image 𝑇𝑖 are fed into the text/image encoder to generate
./reviews/deep_14.md:4511:   379	masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 .
./reviews/deep_14.md:4514:   382	masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 . The loss function can be
./reviews/deep_14.md:4518:   386	text, we construct a knowledge-guided reasoning module, including object-level reasoning, attribute-level reasoning, and relationlevel reasoning. These reasoning processes guide the model to
./reviews/deep_14.md:4519:   387	learn hierarchical semantic local alignments between image and text
./reviews/deep_14.md:4522:   390	visual scene and thus play a crucial role in constructing semantic information. Through object-level reasoning, the model is learning to establish the object-level visual-text connection. Given the paired image 𝐼𝑖
./reviews/deep_14.md:4523:   391	and text 𝑇𝑖 , we ﬁrst obtain all the object nodes 𝑂 = {𝑜𝑖 |𝑖 = 1, 2, … , 𝑛𝑜 }
./reviews/deep_14.md:4524:   392	from the scene graph of text. We randomly select 30 % of object nodes
./reviews/deep_14.md:4527:   395	object masking by masking the sub-sequences in the text that correspond to the selected object nodes 𝑂̂ = {𝑜̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑜 }. Finally, the
./reviews/deep_14.md:4528:   396	object masked text 𝑇̂𝑖𝑜 and image 𝑇𝑖 are fed into the image-text encoder
./reviews/deep_14.md:4529:   397	to obtain masked text features𝑓̂𝑖𝑜,𝑡 and image features 𝑓𝑖𝑣 .
./reviews/deep_14.md:4531:   399	object nodes 𝑂̂ tokens based on the masked text features𝑓̂𝑜,𝑡
./reviews/deep_14.md:4543:   411	between image and text modalities, thereby hindering retrieval performance gains. To tackle this limitation, we propose a Multi-Grained Semantic Alignment (MGA) module to mine discriminative feature representations and perform multi-grained cross-modal alignments. The MAG
./reviews/deep_14.md:4548:   416	from the text encoder as coarse-grained feature representation. Then,
./reviews/deep_14.md:4550:   418	KL divergence between global image-text feature similarity distribution
./reviews/deep_14.md:4579:   447	Given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the attribute pairs 𝑃 = {{𝑜𝑖 , 𝑎𝑖 }|𝑖 = 1, 2, … , 𝑛𝑝 } from the scene graph of text,
./reviews/deep_14.md:4585:   453	masking the sub-sequences of the text that correspond to the selected
./reviews/deep_14.md:4587:   455	the attribute masked text 𝑇̂𝑖𝑎 and image 𝑇𝑖 are fed into the text/image
./reviews/deep_14.md:4588:   456	encoder to generate masked text features𝑓̂𝑖𝑎,𝑡 and image features 𝑓𝑖𝑣 .
./reviews/deep_14.md:4590:   458	nodes {𝑎̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑎 } tokens based on the masked text features𝑓̂𝑖𝑎,𝑡
./reviews/deep_14.md:4640:   508	both image and text modalities using correlation weights. In practice,
./reviews/deep_14.md:4654:   522	the token selection ratio. In terms of textual procedure, we can obtain
./reviews/deep_14.md:4666:   534	the selected local text features in a similar way as 𝑓̂𝑖𝑡 = {𝑡𝑖 𝑡 , 𝑡𝑖 𝑡 , …, 𝑡𝑖 𝑡 },
./reviews/deep_14.md:4671:   539	2021) dataset consists of 20,505 pedestrian images and 41,010 text descriptions with two captions per image. The average length of text descriptions is 23 words. The dataset is divided into three subsets, with
./reviews/deep_14.md:4677:   545	and textual token features 𝑓̂𝑖𝑡 , we perform linear feature transformation
./reviews/deep_14.md:4681:   549	and text embedding 𝑓𝑙𝑜𝑐
./reviews/deep_14.md:4711:   579	list, given a text description as the query. In addition, for a more thorough assessment, we also leverage the mean Average Precision (mAP) as
./reviews/deep_14.md:4721:   589	text feature similarity distributions 𝜓(𝑆(𝑓𝑙𝑜𝑐
./reviews/deep_14.md:4751:   619	We initialized the image and text encoder with pre-trained weights
./reviews/deep_14.md:4758:   626	For text data augmentation, we apply random masking, replacement,
./reviews/deep_14.md:4771:   639	Following the previous work (Zheng et al., 2020), we also adopt an instance loss 𝑖𝑑 that classiﬁes each image and text into speciﬁc classes
./reviews/deep_14.md:4772:   640	based on their corresponding individual IDs, allowing the model to differentiate between image and text groups corresponding to distinct individuals. Above all, the ﬁnal loss function can be formulated as follows:
./reviews/deep_14.md:4779:   647	model under texts (e.g., BERT), to encode the image and text respectively: Dual-Path (Zheng et al., 2020), CMPM/C (Zhang & Lu,
./reviews/deep_14.md:4788:   656	(e.g., CLIP) to encode images and text: CFine (Yan et al., 2023a), TextReID (Han et al., 2021), VGSG He et al. (2023), IRRA (Jiang & Ye,
./reviews/deep_14.md:4799:/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '1020,1350p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:4811:/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '35,555p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:5342:/bin/zsh -lc "pdftotext 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' - | nl -ba | sed -n '480,565p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:5408:   544	spanning group-level context modeling (MACG [5]), classical graph
./reviews/deep_14.md:5442:/bin/zsh -lc "pdftotext 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | nl -ba | sed -n '629,790p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:5530:   715	on Performance: To further analyze the role of textual
./reviews/deep_14.md:5608:/bin/zsh -lc "pdftotext 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | rg -n -i 'ablation|SPG|AHSTG|DFR|Distance|Visualization|heat|response|weight'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:5737:/bin/zsh -lc "pdftotext 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | rg -n -i 'ablation|FHFM|FSM|LFEM|feature hierarchical|feature separation|local feature|visual|heat|t-sne|attention'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:5747:92:    92	of them possess a strong ability to capture global contextual information, their capacity for acquiring ﬁne-grained local features is largely
./reviews/deep_14.md:5762:156:   156	the representation ability of local features by employing a pixel context encoder and a local prototype decoder. Lin et al. (2024) proposed a multi-level relation-aware Transformer (MLRAT), which has
./reviews/deep_14.md:5763:161:   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reviews/deep_14.md:5772:194:   194	proposed an improved YOLOv11 architecture for the automated classiﬁcation of peripheral blood cells, which integrates a Dynamic CrossScale Context Aggregation (DCSCA) module. Through parallel convolution, dynamic attention, and cross-scale interaction, the module enables
./reviews/deep_14.md:5906:/bin/zsh -lc "pdftotext 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '1366,1387p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:5908:/bin/zsh -lc "pdftotext 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '1185,1240p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:5992:/bin/zsh -lc "pdftotext 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '628,895p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:6264:/bin/zsh -lc "pdftotext 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | nl -ba | sed -n '1251,1338p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:6289:  1274	attribute-level reasoning, and relation-level reasoning to establish ﬁnegrained connections between images and text. To further evaluate the
./reviews/deep_14.md:6293:  1278	detailed in IRRA (Jiang & Ye, 2023). Results are presented in Table 6.
./reviews/deep_14.md:6323:  1308	text, we use t-SNE to transform the high-dimensional image and text
./reviews/deep_14.md:6370:/bin/zsh -lc "pdftotext 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '1116,1165p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:6426:/bin/zsh -lc "pdftotext 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | nl -ba | sed -n '787,830p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:6474:/bin/zsh -lc "pdftotext 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '1599,1625p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:6505:/bin/zsh -lc "pdftotext 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | nl -ba | sed -n '1301,1365p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_14.md:6514:  1308	text, we use t-SNE to transform the high-dimensional image and text
./reviews/deep_14.md:6568:  1362	pedestrian embeddings (including both image and text modalities) from the
./reviews/deep_14.md:6570:  1364	The triangles represent text modality embeddings, while the circles represent
./reviews/deep_14.md:6586:3. 机制怎么长出来：机制和重定义绑定很紧。SEEM 负责给伪框和 mask，CLIP 用属性文本给伪身份，单图中同一个人不能出现两次作为去冲突先验；定位噪声用 mask 前景特征约束，类别噪声用置信度降权。
./reviews/deep_14.md:6605:1. 触发观察：作者发现 TIReID 里直接套 MLM 有问题。随机 mask subword 可能遮到功能词或碎片词，训练目标偏离“图文语义对齐”；而行人描述真正有用的是对象、属性和关系，比如“白色上衣”“站在栅栏前”。
./reviews/deep_14.md:6606:2. 重定义动作：它把文本图像对齐从“随机词预测”重定义为“由场景图知识引导的层级语义推理”。关键词是 `blind visual-textual alignment`、`hierarchical scene graph knowledge`、`object-level`、`attribute-level`、`relation-level reasoning`。
./reviews/deep_14.md:6607:3. 机制怎么长出来：机制非常自然。先把文本解析成对象、属性、关系三层场景图，再分别 mask 对象词、属性词、关系词，让图像和剩余文本共同预测被 mask 的语义；再用 MGA 同时做全局句图对齐和局部 token 对齐。
./reviews/deep_14.md:6608:4. 证据闭环：消融把 KGR、MGA、ID loss 拆开。MGA 带来的收益最大，KGR 在多个组合里有稳定增益；还专门比较 MLM、对象推理、属性推理、关系推理和完整 HKGR，完整模型最好。t-SNE、热图、检索可视化用来证明它不是只涨点，而是图文语义空间更靠近。
./reviews/deep_14.md:6609:5. 审稿人为什么买账：这篇卖的是视角。它没有说“我又加了一个语言模块”，而是说“随机 MLM 对行人文本是不合适的，文本本身有层级知识”。这个重定义比模块本身更像贡献。
./reviews/deep_14.md:6633:3. 机制怎么长出来：机制和重定义绑定很紧。SEEM 负责给伪框和 mask，CLIP 用属性文本给伪身份，单图中同一个人不能出现两次作为去冲突先验；定位噪声用 mask 前景特征约束，类别噪声用置信度降权。
./reviews/deep_14.md:6652:1. 触发观察：作者发现 TIReID 里直接套 MLM 有问题。随机 mask subword 可能遮到功能词或碎片词，训练目标偏离“图文语义对齐”；而行人描述真正有用的是对象、属性和关系，比如“白色上衣”“站在栅栏前”。
./reviews/deep_14.md:6653:2. 重定义动作：它把文本图像对齐从“随机词预测”重定义为“由场景图知识引导的层级语义推理”。关键词是 `blind visual-textual alignment`、`hierarchical scene graph knowledge`、`object-level`、`attribute-level`、`relation-level reasoning`。
./reviews/deep_14.md:6654:3. 机制怎么长出来：机制非常自然。先把文本解析成对象、属性、关系三层场景图，再分别 mask 对象词、属性词、关系词，让图像和剩余文本共同预测被 mask 的语义；再用 MGA 同时做全局句图对齐和局部 token 对齐。
./reviews/deep_14.md:6655:4. 证据闭环：消融把 KGR、MGA、ID loss 拆开。MGA 带来的收益最大，KGR 在多个组合里有稳定增益；还专门比较 MLM、对象推理、属性推理、关系推理和完整 HKGR，完整模型最好。t-SNE、热图、检索可视化用来证明它不是只涨点，而是图文语义空间更靠近。
./reviews/deep_14.md:6656:5. 审稿人为什么买账：这篇卖的是视角。它没有说“我又加了一个语言模块”，而是说“随机 MLM 对行人文本是不合适的，文本本身有层级知识”。这个重定义比模块本身更像贡献。
./analyses/deep_8.txt:4:1. 触发观察：作者抓住两个具体问题。第一，TIPR 过去总在做图文“关联”，但文本本身比图像更干净，身份细节更集中，能补足视觉噪声。第二，同一身份的图文正样本并不等价，单视角对应文本是强正样本，跨视角同身份文本可能有外观差异，是弱正样本。
./analyses/deep_8.txt:5:2. 重定义动作：他们把问题从“图文对齐”改写成“跨模态互补协作”，再把正样本从一个类别拆成“强正、弱正、负样本”的有界匹配问题。关键词是 `complementarity`、`virtual-text`、`collaboration`、`identity-bounded matching`。
./analyses/deep_8.txt:6:3. 机制怎么长出来：如果文本有用但推理时图像没有配套文本，就自然推出 `VES`，从图像合成虚拟文本特征；如果虚拟文本要真有用，就要和真实文本特征对齐，再用 `CRCL` 让视觉特征和虚拟文本特征对原型的关系一致。强弱正样本的观察则直接推出 `IBM`，让强正相似度高于上界，弱正落在中间，负样本低于下界。
./analyses/deep_8.txt:7:4. 证据闭环：关键不是总表涨点，而是 `IBM*` 去掉强弱正区分后掉点，证明“正样本内部有层级”这个重定义有意义；`VES` 的各个对齐损失消融证明虚拟文本不是随便加的噪声；检索可视化展示 CoRL 能把视觉相似但文本语义不同的人区分开。
./analyses/deep_8.txt:32:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
./analyses/deep_8.txt:33:2. 重定义动作：作者把 CCReID 从“学习衣服不变视觉特征”改写成“用异质语义描述引导视觉特征摆脱衣服偏置”。关键词是 `semantic guidance`、`visual bottleneck`、`multi-description`、`RGB and gray-scale pseudo-text`、`heterogeneous semantic fusion`。
./analyses/deep_8.txt:34:3. 机制怎么长出来：既然 RGB 文本偏颜色，灰度图可以弱化颜色并突出体型、姿态，就自然推出 `MDG` 同时生成 RGB 和灰度伪文本；既然两种文本各有偏向，就用 `HSF` 同时约束视觉特征；`VSS` 的 ResNet+ViT 融合是补强视觉编码，和核心重定义的绑定不如 MDG/HSF 紧。
./analyses/deep_8.txt:41:3. 机制怎么长出来：如果同一身份的文本 token 应该稳定对齐多个视角图像，就自然推出 `TCL`，约束同一身份多视角图像与文本的相似度方差；如果文本 token 已经学到稳定身份语义，就用 `SFM` 通过文本引导的空间注意力和通道重权重过滤图像 patch。
./analyses/deep_8.txt:42:4. 证据闭环：表 3、表 4 在 AG-ReID 和 CARGO 上分别消融 `TCL`、`SFM` 和组合，且 ViT/ResNet 都有效；更关键的是他们额外画了图文相似度均值和方差，证明 `TCL` 确实降低同一身份跨视角相似度波动；`SFM` 还做了空间、通道、残差和插入层位置消融。
./launch_30deep.sh:18:  PROMPT="你是 ReID 创新方法论拆解员 ${i}。当前目录是论文库,下面这批 PDF 在本目录。用 \`pdftotext -f 1 -l 8 '文件名' -\` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_22.md:14:你是 ReID 创新方法论拆解员 22。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_22.md:36:我会先确认目录里的文件和项目规则能读到哪些，再逐篇用 `pdftotext -f 1 -l 8` 抽方法部分。这个任务偏文献方法论拆解，我会把重点放在“观察如何被包装成问题”和“证据如何支撑视角”，不是复述模块结构。
./reviews/deep_22.md:92:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_22.md:105:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_22.md:132:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_22.md:139:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_22.md:229:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_22.md:230:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_22.md:240:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_22.md:247:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_22.md:260:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_22.md:264:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_22.md:308:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_22.md:315:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_22.md:358:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_22.md:371:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_22.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_22.md:495:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_22.md:516:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_22.md:532:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_22.md:590:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_22.md:591:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_22.md:617:项目目标仍然是先读文献、做差距分析，不开实验。现在我开始按你给的六篇逐篇抽取前 8 页文本，并定位方法章节和消融证据章节。
./reviews/deep_22.md:625:/bin/zsh -lc "pdftotext -f 1 -l 8 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:636:that seeks to match pedestrian images with their corresponding textual descriptions. A key challenge in this task arises
./reviews/deep_22.md:639:embedding methods, which map images and texts to fixed
./reviews/deep_22.md:643:pedestrian images and text as probabilistic distributions and
./reviews/deep_22.md:646:Representation Modeling (DRM) encodes images and text into
./reviews/deep_22.md:649:distributions of text and masked text with their associated image
./reviews/deep_22.md:689:Fig. 1. Problems with existing TBPR: (a) TBPR faces complex image–text
./reviews/deep_22.md:702:instance-specific. Consequently, a single text description theoretically corresponds to a “set” of valid visual appearances,
./reviews/deep_22.md:709:limitations. First, by mapping images and texts to fixed points
./reviews/deep_22.md:711:of TBPR, where a general text should match multiple valid
./reviews/deep_22.md:717:1051-8215 © 2026 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
./reviews/deep_22.md:728:ignore the logical partial-order relationship: a general text
./reviews/deep_22.md:735:both pedestrian images and textual descriptions as Gaussian
./reviews/deep_22.md:739:which aligns visual and textual distributions, and probabilistic containment, which imposes containment constraints
./reviews/deep_22.md:742:(DRM) module that encodes pedestrian images and texts into
./reviews/deep_22.md:751:enforces that the Gaussian distribution of a textual description
./reviews/deep_22.md:771:between general text descriptions and specific visual
./reviews/deep_22.md:782:Li et al. [1] introduced text-based person retrieval (TBPR)
./reviews/deep_22.md:785:pretrained feature extractors, such as VGG [14] or ResNet50/101 [15] for images and LSTM [16] or BERT [17] for text
./reviews/deep_22.md:792:local visual-textual alignment. During this phase, datasets
./reviews/deep_22.md:793:such as ICFG-PEDES and RSTPReid were also proposed
./reviews/deep_22.md:796:recent approaches have shifted toward unified pretrained backbones that jointly model image-text interactions at scale.
./reviews/deep_22.md:797:For instance, IRRA [5] leverages masked language modeling to introduce implicit semantic reasoning, while CFine
./reviews/deep_22.md:801:RDE [10] addresses noisy image-text correspondences using
./reviews/deep_22.md:807:to build LUPerson-T by generating pseudo-text descriptions
./reviews/deep_22.md:808:from the large-scale LUPerson dataset [28]. APTM [29]
./reviews/deep_22.md:812:to augment LUPerson with more diverse and robust text
./reviews/deep_22.md:816:LLMs for high-quality textual annotations. UFineBench [33]
./reviews/deep_22.md:828:both images and textual descriptions as probabilistic Gaussian
./reviews/deep_22.md:842:incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
./reviews/deep_22.md:861:way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
./reviews/deep_22.md:880:learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
./reviews/deep_22.md:886:text features to explicitly capture complex semantic inclusion
./reviews/deep_22.md:897:representations for images and texts, followed by a detailed
./reviews/deep_22.md:903:initialize PDA with the full CLIP image and text encoder
./reviews/deep_22.md:904:where the image encoder and text encoder are both 12-layer
./reviews/deep_22.md:924:2) Text Encoder: For an input textual description T , we
./reviews/deep_22.md:925:employ the CLIP text encoder to extract its features. Following
./reviews/deep_22.md:926:CLIP and IRRA, the text is first tokenized using lower-cased
./reviews/deep_22.md:934:Fig. 2. The framework of PDA. PDA models image and text as well as their mask forms as gaussian distributions, then narrows the distance between image
./reviews/deep_22.md:935:and text probabilistic representations while exploring the inclusion relationship within and between modalities.
./reviews/deep_22.md:946:projection to obtain the final Euclidean text representations.
./reviews/deep_22.md:947:Similarly, for the j-th text input, we denote its representation
./reviews/deep_22.md:956:text feature and { f1 , . . ., f M } correspond to local token-level
./reviews/deep_22.md:961:image and text features into Gaussian distributions. This
./reviews/deep_22.md:971:The derivation of the visual and textual probabilistic representations is as Fig 3. For the i-th image, we start with
./reviews/deep_22.md:977:Fig. 3. The derivation of the visual and textual probabilistic representations
./reviews/deep_22.md:985:Similarly, for the j-th text, f0 j from fTj is processed to compute
./reviews/deep_22.md:989:Having modeled the image and text features as Gaussian
./reviews/deep_22.md:994:image-text pairs, and large for non-matching or ambiguous
./reviews/deep_22.md:1005:ensures that the visual-textual similarity is modulated by
./reviews/deep_22.md:1018:texts with low uncertainty is primarily determined by the
./reviews/deep_22.md:1020:while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./reviews/deep_22.md:1021:LDRM to align image-text pairs:
./reviews/deep_22.md:1032:where qi j ∈ {0, 1} indicates whether the image-text pair
./reviews/deep_22.md:1045:In pedestrian retrieval, text descriptions such as “a person in a jacket” are expected to generalize across multiple
./reviews/deep_22.md:1048:noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
./reviews/deep_22.md:1051:Given a positive image–text pair (Ii , T j ), we model their
./reviews/deep_22.md:1060:scope. To reflect the intuition that textual descriptions are
./reviews/deep_22.md:1062:is contained within the text distribution Zt j . The log-inclusion
./reviews/deep_22.md:1078:the text distribution. Intuitively, the first term is large when pvi
./reviews/deep_22.md:1087:distribution is effectively captured by the text distribution.
./reviews/deep_22.md:1097:between the sharpened image distribution and the text distribution. Formally, our containment objective acts as a
./reviews/deep_22.md:1100:where the text distribution serves as a “probabilistic envelope”
./reviews/deep_22.md:1248:masked text feature distribution. By combining both complete
./reviews/deep_22.md:1249:and masked text distributions, CMC encourages the image
./reviews/deep_22.md:1250:representation to align with a semantically broader textual
./reviews/deep_22.md:1252:textual cues are partially missing.
./reviews/deep_22.md:1253:Although the assumption that the text distribution encompasses the image distribution may not strictly hold in all
./reviews/deep_22.md:1256:the degree of containment rather than enforcing strict inclusion. This design ensures stable gradients even when textual
./reviews/deep_22.md:1258:contextual information. Furthermore, since both image and text
./reviews/deep_22.md:1277:text,
./reviews/deep_22.md:1310:Zvi and text Zt j are contained within their corresponding
./reviews/deep_22.md:1331:the cross-modal robustness to incomplete text descriptions
./reviews/deep_22.md:1333:distribution Zvi with the masked text distribution Zmask
./reviews/deep_22.md:1336:introduces uncertainty by randomly masking textual tokens.
./reviews/deep_22.md:1341:or masked text due to the reduced visual information and
./reviews/deep_22.md:1358:positive image-text pairs (Ii , T j ), as emphasized in the main
./reviews/deep_22.md:1359:text. In real-world TBPR scenarios, some images (e.g., I sp ) are
./reviews/deep_22.md:1360:generally well-described by the accompanying text, making
./reviews/deep_22.md:1376:of the text than the image and LIMC enhance the model’s
./reviews/deep_22.md:1386:ICFG-PEDES [21], RSTPReid [54] and adopt the Rank-k and
./reviews/deep_22.md:1398:detailed identity focused text. The dataset divides into a
./reviews/deep_22.md:1412:• RSTPReid [54] features 20,505 images across 4,101
./reviews/deep_22.md:1422:given a text description as a query, at least one matching
./reviews/deep_22.md:1443:PDA method across three widely used datasets: CUHKPEDES, ICFG-PEDES, and RSTPReid.
./reviews/deep_22.md:1451:(MLLMs) for diverse textual generation, PDA improves Rank1 by 1.25% and mAP by 0.01%. When compared with MUM,
./reviews/deep_22.md:1455:relative to APTM, a computationally expensive framework
./reviews/deep_22.md:1469:M ETHODS ON RSTPR EID DATASET
./reviews/deep_22.md:1478:Additionally, when compared with APTM, the strongest non
./reviews/deep_22.md:1483:3) Performance Comparisons on RSTPReid: To further
./reviews/deep_22.md:1485:the RSTPReid dataset, with results summarized in Table III.
./reviews/deep_22.md:1492:margins over APTM are also significant, with PDA improving
./reviews/deep_22.md:1502:datasets demonstrates PDA’s strong generalization capability and practical potential for real-world text-based person
./reviews/deep_22.md:1513:text-based person retrieval. The first protocol evaluates the
./reviews/deep_22.md:1523:baselines. For IRRA, RDE, and NAM, we manually reproduce
./reviews/deep_22.md:1533:/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:2347:context of cross-modality person re-identification.
./reviews/deep_22.md:2487:/bin/zsh -lc "pdftotext -f 1 -l 8 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:3310:/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:4837:attribute-guided prompt for unsupervised text-based person retrieval. In Proceedings of the International Joint
./reviews/deep_22.md:4867:/bin/zsh -lc "pdftotext -f 1 -l 8 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:4892:(ESFA) module, which supplements textual descriptions for cross-modality images and aligns
./reviews/deep_22.md:4893:image-text pairs within each modality, alleviating
./reviews/deep_22.md:4896:the similarity between intra-modality image-text
./reviews/deep_22.md:4897:pairs to be consistent with that between intermodality text-text pairs, indirectly aligning visual
./reviews/deep_22.md:4898:features with cross-modality semantics. Furthermore, we design a Cross-View Semantics Compensation (CVSC) module, which integrates multiview texts and improves the image-text matching of
./reviews/deep_22.md:4937:the more effective alignment of visual features than learnable textual
./reviews/deep_22.md:4939:of images to inter-modality texts challenging. (iii) Multi-view texts
./reviews/deep_22.md:4983:Learning textual prompts [Zhou et al., 2022] effectively addresses this issue, as illustrated in Figure 1(a), but it still
./reviews/deep_22.md:4992:This inspires us to supplement specific texts with the assistance of LLaVA and align image-text pairs within each
./reviews/deep_22.md:4993:modality. Furthermore, the alignment of images to intermodality texts is also necessary as it can further alleviate the
./reviews/deep_22.md:4997:an indirect alignment of images to inter-modality texts. In
./reviews/deep_22.md:5002:the alignment of visual features is beneficial for further enhancing their modality invariance. This prompts us to enrich pedestrian semantics with multi-view texts.
./reviews/deep_22.md:5006:Cross-View Semantics Compensation (CVSC). ESFA leverages LLaVA to generate textual descriptions for visible and
./reviews/deep_22.md:5008:between visible (infrared) image-text pairs to align crossmodality visual features into the semantic space. CSIA constrains the similarity between intra-modality image-text pairs
./reviews/deep_22.md:5009:to be consistent with that between inter-modality text-text
./reviews/deep_22.md:5010:pairs, achieving the indirect alignment of visible visual features with infrared semantics as well as infrared visual features with visible semantics. CVSC integrates text features
./reviews/deep_22.md:5011:from another view into the current view and accordingly improves the image-text matching in ESFA and CSIA from oneto-one to one-to-many, thereby further advancing their alignment. Our RSBA is trained end-to-end, with only the visual
./reviews/deep_22.md:5021:alignment between images and texts, further strengthening the modality invariance of visual features.
./reviews/deep_22.md:5271:employed CLIP to generate local textual prototypes for mining discriminative part features. In VIReID, CSDN [Yu et al.,
./reviews/deep_22.md:5272:2025] incorporated trainable textual prompts to acquire implicit pedestrian descriptions, aligning visual features of visible and infrared images into the semantic space. However, the
./reviews/deep_22.md:5369:ESFA, which leverages LLaVA to generate explicit textual
./reviews/deep_22.md:5374:clearer and more detailed explicit semantics, such as age, gender, and clothing, compared to the learnable textual prompt
./reviews/deep_22.md:5401:and input them into the textual encoder to extract feair N
./reviews/deep_22.md:5411:The above two alignments are based on the one-to-one matching between image and text. However, within each modality,
./reviews/deep_22.md:5417:semantics derived solely from single-view text are one-sided
./reviews/deep_22.md:5419:address this limitation, we design CVSC to explore the oneto-many correspondence between images and texts.
./reviews/deep_22.md:5420:As illustrated in Figure 2, we introduce an attention fusion module to integrate information in the textual feature
./reviews/deep_22.md:5421:from another view into the textual feature of the current view.
./reviews/deep_22.md:5422:Specifically, for the visible textual feature tvis
./reviews/deep_22.md:5424:select a textual feature tvis
./reviews/deep_22.md:5541:m,i represents the multi-view textual feature corresponding to livis ,
./reviews/deep_22.md:5545:can acquire the multi-view infrared textual feature tir
./reviews/deep_22.md:5553:ESFA achieves the alignment of images and texts in each
./reviews/deep_22.md:5555:inter-modality texts. A straightforward approach is to maximize the similarity between them similar to the above process. However, cross-modality texts describe the same object
./reviews/deep_22.md:5556:with conflicting attributes due to visual ambiguity. For example, the clothing exhibits ’blue’ in the visible text while being
./reviews/deep_22.md:5558:of similarity between images and inter-modality texts may
./reviews/deep_22.md:5561:CSIA constrains its similarity with the visible text feature tvis
./reviews/deep_22.md:5563:to be equal to the similarity between the infrared text feature
./reviews/deep_22.md:5566:i and visible text feature ti , thereby indirectly establishing
./reviews/deep_22.md:5569:infrared visual features fiir and visible text features tvis
./reviews/deep_22.md:5648:images and texts within each modality. In addition, we also
./reviews/deep_22.md:5649:redefine the Equation (8) to the following Equation (14), indirectly aligning images with multi-view inter-modality texts:
./reviews/deep_22.md:5683:This similarity consistency loss not only achieves the alignment of images with inter-modality texts but also indirectly
./reviews/deep_22.md:5684:maximizes the similarity between infrared and visible texts,
./reviews/deep_22.md:5746:our framework. During inference, the textual encoder and
./reviews/deep_22.md:5785:visual encoder and a textual encoder, with ResNet50 [He et
./reviews/deep_22.md:5794:to 3e-4 for the visual encoder and 1e-6 for the textual encoder
./reviews/deep_22.md:5808:Effectiveness of CSIA. CSIA constrains the consistent similarity between intra-modality image-text pairs and intermodality text-text pairs to establish the correspondence between images and cross-modality texts. As detailed in Table
./reviews/deep_22.md:6104:ment of inter-modality image-text pairs plays a positive role
./reviews/deep_22.md:6106:Effectiveness of CVSC. CVSC integrates multi-view texts to
./reviews/deep_22.md:6119:former optimizes the model to align image-text pairs within
./reviews/deep_22.md:6121:correspondence between image-text pairs across modalities.
./reviews/deep_22.md:6156:from image-text pairs within each modality, which naturally
./reviews/deep_22.md:6157:correspond to each other, images and inter-modality texts are
./reviews/deep_22.md:6160:between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
./reviews/deep_22.md:6180:The number of cross-view texts
./reviews/deep_22.md:6182:multi-view texts, and we achieve this by integrating text with
./reviews/deep_22.md:6184:with texts from multiple additional views. However, we observe that the recognition performance degrades as the number of views increases (2 and 3). This is because the generated
./reviews/deep_22.md:6245:the enrichment of pedestrian semantics with multi-view texts,
./reviews/deep_22.md:6246:while we ignore the quality of texts, the number of cross-view
./reviews/deep_22.md:6247:texts, and the strategy of text fusion, which all affect the richness of the semantics. These limitations motivate us to explore the semantics of diversity and richness more deeply.
./reviews/deep_22.md:6274:the correspondence of image-text pairs, aligning visual features into the semantic space. CSIA introduces the similarity
./reviews/deep_22.md:6299:[Li et al., 2023] Siyuan Li, Li Sun, and Qingli Li. Clipreid: exploiting vision-language model for image reidentification without concrete text labels. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 1405–1413, 2023.
./reviews/deep_22.md:6374:/bin/zsh -lc "pdftotext -f 1 -l 8 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:6379:contextual relationship-mining graph (ICRMG) and text contextual relationship-mining graph (TCRMG), which encode
./reviews/deep_22.md:6380:the mutual semantic relationships between image and text local
./reviews/deep_22.md:6382:features by aggregating their semantic contextual information
./reviews/deep_22.md:6385:as the final image and text feature expression to calculate
./reviews/deep_22.md:6393:directly determined whether the image and text are the same
./reviews/deep_22.md:6396:1) Image Contextual Relationship-Mining Graph: In the
./reviews/deep_22.md:6412:this end, we design the Image Contextual Relationship-Mining
./reviews/deep_22.md:6423:Fig. 2. (a) shows the architecture of RMGNet, while (b) and (c) represent the architectures of ICRMG and TCRMG, respectively. Given N image-text pairs
./reviews/deep_22.md:6425:relationships among different local features by the ICRMG and the TCRMG. This can generate the relationship-enhanced image and text local features, FrIel
./reviews/deep_22.md:6454:textual relationship-mining graph denoted as G I = H I , E I ,
./reviews/deep_22.md:6517:2) Text Contextual Relationship-Mining Graph: After
./reviews/deep_22.md:6518:obtaining the relationship-enhanced person image local features, in order to obtain the relationship-enhanced person text local features. We design the text contextual
./reviews/deep_22.md:6520:Since text is a type of data with sequential attributes, its
./reviews/deep_22.md:6527:matching. For example, the text description ‘A woman was
./reviews/deep_22.md:6533:image-text, that the person images corresponding to the two
./reviews/deep_22.md:6537:bag’. This can effectively distinguish the description text of
./reviews/deep_22.md:6538:different person images. Therefore, we model the interrelationship of person text local features from both forward and
./reviews/deep_22.md:6541:long short-term memory (Bi-LSTM) network [46] for text
./reviews/deep_22.md:6543:the forward text contextual relationship-mining sub-graph
./reviews/deep_22.md:6544:(FTCRMSG) and the backward text contextual relationshipmining sub-graph (BTCRMSG). The two sub-graphs model
./reviews/deep_22.md:6545:and learn the correlation of the text local features from the
./reviews/deep_22.md:6547:we first input the description text into the BERT model in forfD
./reviews/deep_22.md:6548:ward sequence to learn the forward text local
./reviews/deep_22.md:6568:by inputting person description text in reverse, we can obtain
./reviews/deep_22.md:6569:backward text local features FlbD , as follows:
./reviews/deep_22.md:6574:Second, we employ GNN to model and mine the interrelationships between person text local features. We need to
./reviews/deep_22.md:6579:of forward person text local features, that is, the nodes in
./reviews/deep_22.md:6606:text local features, ei, j represents the mutual semantic rela-
./reviews/deep_22.md:6625:the interrelationship between different text local features as
./reviews/deep_22.md:6644:where frli represents the updated forward text local feature
./reviews/deep_22.md:6646:Therefore, the relationship guide updated backward text local
./reviews/deep_22.md:6682:ward relationship-enhanced person text local features, Fr el
./reviews/deep_22.md:6685:relationship-enhanced text local features. Finally, we weighted
./reviews/deep_22.md:6686:and fuse the forward and backward relationship-enhanced text
./reviews/deep_22.md:6687:local features to generate the final text relationship enhancement local features FrDel as:
./reviews/deep_22.md:6737:image and text relationship-enhanced features,
./reviews/deep_22.md:6809:whether the person image matches the description text.
./reviews/deep_22.md:6816:CMPM loss to optimize the learning of person image and text
./reviews/deep_22.md:6829:feature, F jD is the jth person text feature, that is generated
./reviews/deep_22.md:6846:where F jD represents the normalized text features, and FiI D F jD
./reviews/deep_22.md:6847:represents the projection of the image features in the text
./reviews/deep_22.md:6849:texts in a batch of training samples. Therefore, the normalized
./reviews/deep_22.md:6860:divergence of the image-to-text projection probability and the
./reviews/deep_22.md:6861:true matching probability, the final image-to-text matching loss
./reviews/deep_22.md:6876:After that, the image and text features in eq:15 also need
./reviews/deep_22.md:6877:to be exchanged to calculate the text-to-image matching loss.
./reviews/deep_22.md:6997:In order to accurately match the person image and text,
./reviews/deep_22.md:7002:final person text features F D can be obtained. After that,
./reviews/deep_22.md:7013:CUHK-PEDES [26], ICFG-PEDES [39], and RSTPReid [31],
./reviews/deep_22.md:7019:40206 images of different person and 80440 description texts.
./reviews/deep_22.md:7021:images, and 68108 person description texts. The verification
./reviews/deep_22.md:7022:set includes 2000 person, 3078 person images, and 6156 person descriptions texts.
./reviews/deep_22.md:7025:and 54522 image-text pairs, and the description text contains
./reviews/deep_22.md:7027:training and test sets. The training set includes 34674 imagetext pairs of 3102 person, while the test set consists of 1,000
./reviews/deep_22.md:7028:person,19,848 images, and their corresponding text descriptions.
./reviews/deep_22.md:7029:3) RSTPReid: Person images in this dataset are gathered
./reviews/deep_22.md:7033:and 41010 corresponding text descriptions. The training set
./reviews/deep_22.md:7034:includes 3701 people, 18505 images, and 37010 text descriptions corresponding to images. The validation set consists of
./reviews/deep_22.md:7035:200 people, 1000 images, and 2000 text descriptions.
./reviews/deep_22.md:7040:and text features, respectively. We set the input image size of
./reviews/deep_22.md:7041:the network to 384 ∗ 128, and each text length is unified to
./reviews/deep_22.md:7042:64. The dimensions of image and text features are set to 512.
./reviews/deep_22.md:7044:random cropping. Each training batch consists of 32 imagetext pairs and a total of 80 epochs of iterative training are
./reviews/deep_22.md:7082:utilized ViT and BERT as feature extractors. During experiments, we match person images and texts by computing the
./reviews/deep_22.md:7097:of text local feature relationships on the TI-ReID performance.
./reviews/deep_22.md:7098:An analysis of the text contextual relationship-mining graph
./reviews/deep_22.md:7100:using only forward text local features (F-TCRMG), (b) the
./reviews/deep_22.md:7102:text local features (B-TCRMG), and (c) the relational learning
./reviews/deep_22.md:7104:backward text local features, i.e. the method used in this paper.
./reviews/deep_22.md:7109:that the best effect is achieved when jointly using the text
./reviews/deep_22.md:7112:that although the Bi-LSTM has learned some reverse text
./reviews/deep_22.md:7114:obtained by inputting text in reverse. Furthermore, when
./reviews/deep_22.md:7116:using backward text local features exclusively, the absence of
./reviews/deep_22.md:7117:supplementation and guidance from forward text local features
./reviews/deep_22.md:7121:and enhances the distinguishability of person text features by
./reviews/deep_22.md:7122:jointly utilizing forward and backward text local features.
./reviews/deep_22.md:7147:person image and text features respectively. In the experiment, the extracted features are directly used for similarity
./reviews/deep_22.md:7148:calculation to match person images and texts. Table IV shows
./reviews/deep_22.md:7157:RSBA 把“语义无模态差异”做成主视角，但它又没有停在“用文本监督图像”这个老说法上，而是专门处理了跨模态文本语义冲突和多视角文本不完整这两个阻碍。RMGNet 的触发点则是局部对齐已经不够，真正能区分相似行人的是局部之间的关系，例如“背着包”和“穿着包”的差别。
./reviews/deep_22.md:7175:/bin/zsh -lc "pdftotext -f 1 -l 8 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | rg -n \"III\\. M|A\\. Feature|B\\. Distribution|C\\. Cross|D\\. Intra|E\\. Training|IV\\. E|Ablation|Visualization|Containment|DRM|CMC|IMC\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:7177:20:    20	Representation Modeling (DRM) encodes images and text into
./reviews/deep_22.md:7180:116:   116	(DRM) module that encodes pedestrian images and texts into
./reviews/deep_22.md:7193:395:   395	LDRM to align image-text pairs:
./reviews/deep_22.md:7198:422:   422	noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
./reviews/deep_22.md:7201:623:   623	and masked text distributions, CMC encourages the image
./reviews/deep_22.md:7217:750:   750	of the text than the image and LIMC enhance the model’s
./reviews/deep_22.md:7221:/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | rg -n \"3\\.3|3\\.4|3\\.5|Adaptive Parameter|θ|fused|Ablation|Table 3|Visualization|PKP|APE\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:7223:/bin/zsh -lc "pdftotext -f 1 -l 8 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -n \"3\\.2|3\\.3|3\\.4|Consistent|Cross-View|Lo2m|Lsc|Equation|Ablation|Further Discussions|Limitations|Table 4|Table 5\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:7289:30:    30	features with cross-modality semantics. Furthermore, we design a Cross-View Semantics Compensation (CVSC) module, which integrates multiview texts and improves the image-text matching of
./reviews/deep_22.md:7291:138:   138	Cross-View Semantics Compensation (CVSC). ESFA leverages LLaVA to generate textual descriptions for visible and
./reviews/deep_22.md:7304:781:   781	redefine the Equation (8) to the following Equation (14), indirectly aligning images with multi-view inter-modality texts:
./reviews/deep_22.md:7327:1292:  1292	between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
./reviews/deep_22.md:7338:/bin/zsh -lc "pdftotext -f 1 -l 8 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | rg -n \"III\\. M|A\\. Overview|B\\. Intra|C\\. Cross|D\\. Training|E\\. Similarity|ICRMG|TCRMG|CMRM|C2FL|Acc|Ablation\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:7341:26:    26	inherent in the image and text features. The CMRM module
./reviews/deep_22.md:7354:280:   280	contextual relationship-mining graph (ICRMG) and text contextual relationship-mining graph (TCRMG), which encode
./reviews/deep_22.md:7360:324:   324	Fig. 2. (a) shows the architecture of RMGNet, while (b) and (c) represent the architectures of ICRMG and TCRMG, respectively. Given N image-text pairs
./reviews/deep_22.md:7361:326:   326	relationships among different local features by the ICRMG and the TCRMG. This can generate the relationship-enhanced image and text local features, FrIel
./reviews/deep_22.md:7386:1001:  1001	using only forward text local features (F-TCRMG), (b) the
./reviews/deep_22.md:7387:1003:  1003	text local features (B-TCRMG), and (c) the relational learning
./reviews/deep_22.md:7417:/bin/zsh -lc "pdftotext -f 1 -l 8 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '460,875p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:7460:   501	ESFA, which leverages LLaVA to generate explicit textual
./reviews/deep_22.md:7465:   506	clearer and more detailed explicit semantics, such as age, gender, and clothing, compared to the learnable textual prompt
./reviews/deep_22.md:7492:   533	and input them into the textual encoder to extract feair N
./reviews/deep_22.md:7502:   543	The above two alignments are based on the one-to-one matching between image and text. However, within each modality,
./reviews/deep_22.md:7508:   549	semantics derived solely from single-view text are one-sided
./reviews/deep_22.md:7510:   551	address this limitation, we design CVSC to explore the oneto-many correspondence between images and texts.
./reviews/deep_22.md:7511:   552	As illustrated in Figure 2, we introduce an attention fusion module to integrate information in the textual feature
./reviews/deep_22.md:7512:   553	from another view into the textual feature of the current view.
./reviews/deep_22.md:7513:   554	Specifically, for the visible textual feature tvis
./reviews/deep_22.md:7515:   556	select a textual feature tvis
./reviews/deep_22.md:7632:   673	m,i represents the multi-view textual feature corresponding to livis ,
./reviews/deep_22.md:7636:   677	can acquire the multi-view infrared textual feature tir
./reviews/deep_22.md:7644:   685	ESFA achieves the alignment of images and texts in each
./reviews/deep_22.md:7646:   687	inter-modality texts. A straightforward approach is to maximize the similarity between them similar to the above process. However, cross-modality texts describe the same object
./reviews/deep_22.md:7647:   688	with conflicting attributes due to visual ambiguity. For example, the clothing exhibits ’blue’ in the visible text while being
./reviews/deep_22.md:7649:   690	of similarity between images and inter-modality texts may
./reviews/deep_22.md:7652:   693	CSIA constrains its similarity with the visible text feature tvis
./reviews/deep_22.md:7654:   695	to be equal to the similarity between the infrared text feature
./reviews/deep_22.md:7657:   698	i and visible text feature ti , thereby indirectly establishing
./reviews/deep_22.md:7660:   701	infrared visual features fiir and visible text features tvis
./reviews/deep_22.md:7739:   780	images and texts within each modality. In addition, we also
./reviews/deep_22.md:7740:   781	redefine the Equation (8) to the following Equation (14), indirectly aligning images with multi-view inter-modality texts:
./reviews/deep_22.md:7774:   815	This similarity consistency loss not only achieves the alignment of images with inter-modality texts but also indirectly
./reviews/deep_22.md:7775:   816	maximizes the similarity between infrared and visible texts,
./reviews/deep_22.md:7837:/bin/zsh -lc "pdftotext -f 1 -l 8 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '320,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:7839:   320	projection to obtain the final Euclidean text representations.
./reviews/deep_22.md:7840:   321	Similarly, for the j-th text input, we denote its representation
./reviews/deep_22.md:7849:   330	text feature and { f1 , . . ., f M } correspond to local token-level
./reviews/deep_22.md:7854:   335	image and text features into Gaussian distributions. This
./reviews/deep_22.md:7864:   345	The derivation of the visual and textual probabilistic representations is as Fig 3. For the i-th image, we start with
./reviews/deep_22.md:7870:   351	Fig. 3. The derivation of the visual and textual probabilistic representations
./reviews/deep_22.md:7878:   359	Similarly, for the j-th text, f0 j from fTj is processed to compute
./reviews/deep_22.md:7882:   363	Having modeled the image and text features as Gaussian
./reviews/deep_22.md:7887:   368	image-text pairs, and large for non-matching or ambiguous
./reviews/deep_22.md:7898:   379	ensures that the visual-textual similarity is modulated by
./reviews/deep_22.md:7911:   392	texts with low uncertainty is primarily determined by the
./reviews/deep_22.md:7913:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./reviews/deep_22.md:7914:   395	LDRM to align image-text pairs:
./reviews/deep_22.md:7925:   406	where qi j ∈ {0, 1} indicates whether the image-text pair
./reviews/deep_22.md:7938:   419	In pedestrian retrieval, text descriptions such as “a person in a jacket” are expected to generalize across multiple
./reviews/deep_22.md:7941:   422	noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
./reviews/deep_22.md:7944:   425	Given a positive image–text pair (Ii , T j ), we model their
./reviews/deep_22.md:7953:   434	scope. To reflect the intuition that textual descriptions are
./reviews/deep_22.md:7955:   436	is contained within the text distribution Zt j . The log-inclusion
./reviews/deep_22.md:7971:   452	the text distribution. Intuitively, the first term is large when pvi
./reviews/deep_22.md:7980:   461	distribution is effectively captured by the text distribution.
./reviews/deep_22.md:7990:   471	between the sharpened image distribution and the text distribution. Formally, our containment objective acts as a
./reviews/deep_22.md:7993:   474	where the text distribution serves as a “probabilistic envelope”
./reviews/deep_22.md:8141:   622	masked text feature distribution. By combining both complete
./reviews/deep_22.md:8142:   623	and masked text distributions, CMC encourages the image
./reviews/deep_22.md:8143:   624	representation to align with a semantically broader textual
./reviews/deep_22.md:8145:   626	textual cues are partially missing.
./reviews/deep_22.md:8146:   627	Although the assumption that the text distribution encompasses the image distribution may not strictly hold in all
./reviews/deep_22.md:8149:   630	the degree of containment rather than enforcing strict inclusion. This design ensures stable gradients even when textual
./reviews/deep_22.md:8151:   632	contextual information. Furthermore, since both image and text
./reviews/deep_22.md:8170:   651	text,
./reviews/deep_22.md:8203:   684	Zvi and text Zt j are contained within their corresponding
./reviews/deep_22.md:8224:   705	the cross-modal robustness to incomplete text descriptions
./reviews/deep_22.md:8226:   707	distribution Zvi with the masked text distribution Zmask
./reviews/deep_22.md:8229:   710	introduces uncertainty by randomly masking textual tokens.
./reviews/deep_22.md:8234:   715	or masked text due to the reduced visual information and
./reviews/deep_22.md:8251:   732	positive image-text pairs (Ii , T j ), as emphasized in the main
./reviews/deep_22.md:8252:   733	text. In real-world TBPR scenarios, some images (e.g., I sp ) are
./reviews/deep_22.md:8253:   734	generally well-described by the accompanying text, making
./reviews/deep_22.md:8269:   750	of the text than the image and LIMC enhance the model’s
./reviews/deep_22.md:8279:   760	ICFG-PEDES [21], RSTPReid [54] and adopt the Rank-k and
./reviews/deep_22.md:8282:/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | sed -n '260,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:8547:/bin/zsh -lc "pdftotext -f 1 -l 8 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '296,910p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:8550:   297	1) Image Contextual Relationship-Mining Graph: In the
./reviews/deep_22.md:8566:   313	this end, we design the Image Contextual Relationship-Mining
./reviews/deep_22.md:8577:   324	Fig. 2. (a) shows the architecture of RMGNet, while (b) and (c) represent the architectures of ICRMG and TCRMG, respectively. Given N image-text pairs
./reviews/deep_22.md:8579:   326	relationships among different local features by the ICRMG and the TCRMG. This can generate the relationship-enhanced image and text local features, FrIel
./reviews/deep_22.md:8608:   355	textual relationship-mining graph denoted as G I = H I , E I ,
./reviews/deep_22.md:8671:   418	2) Text Contextual Relationship-Mining Graph: After
./reviews/deep_22.md:8672:   419	obtaining the relationship-enhanced person image local features, in order to obtain the relationship-enhanced person text local features. We design the text contextual
./reviews/deep_22.md:8674:   421	Since text is a type of data with sequential attributes, its
./reviews/deep_22.md:8681:   428	matching. For example, the text description ‘A woman was
./reviews/deep_22.md:8687:   434	image-text, that the person images corresponding to the two
./reviews/deep_22.md:8691:   438	bag’. This can effectively distinguish the description text of
./reviews/deep_22.md:8692:   439	different person images. Therefore, we model the interrelationship of person text local features from both forward and
./reviews/deep_22.md:8695:   442	long short-term memory (Bi-LSTM) network [46] for text
./reviews/deep_22.md:8697:   444	the forward text contextual relationship-mining sub-graph
./reviews/deep_22.md:8698:   445	(FTCRMSG) and the backward text contextual relationshipmining sub-graph (BTCRMSG). The two sub-graphs model
./reviews/deep_22.md:8699:   446	and learn the correlation of the text local features from the
./reviews/deep_22.md:8701:   448	we first input the description text into the BERT model in forfD
./reviews/deep_22.md:8702:   449	ward sequence to learn the forward text local
./reviews/deep_22.md:8722:   469	by inputting person description text in reverse, we can obtain
./reviews/deep_22.md:8723:   470	backward text local features FlbD , as follows:
./reviews/deep_22.md:8728:   475	Second, we employ GNN to model and mine the interrelationships between person text local features. We need to
./reviews/deep_22.md:8733:   480	of forward person text local features, that is, the nodes in
./reviews/deep_22.md:8760:   507	text local features, ei, j represents the mutual semantic rela-
./reviews/deep_22.md:8779:   526	the interrelationship between different text local features as
./reviews/deep_22.md:8798:   545	where frli represents the updated forward text local feature
./reviews/deep_22.md:8800:   547	Therefore, the relationship guide updated backward text local
./reviews/deep_22.md:8836:   583	ward relationship-enhanced person text local features, Fr el
./reviews/deep_22.md:8839:   586	relationship-enhanced text local features. Finally, we weighted
./reviews/deep_22.md:8840:   587	and fuse the forward and backward relationship-enhanced text
./reviews/deep_22.md:8841:   588	local features to generate the final text relationship enhancement local features FrDel as:
./reviews/deep_22.md:8891:   638	image and text relationship-enhanced features,
./reviews/deep_22.md:8963:   710	whether the person image matches the description text.
./reviews/deep_22.md:8970:   717	CMPM loss to optimize the learning of person image and text
./reviews/deep_22.md:8983:   730	feature, F jD is the jth person text feature, that is generated
./reviews/deep_22.md:9000:   747	where F jD represents the normalized text features, and FiI D F jD
./reviews/deep_22.md:9001:   748	represents the projection of the image features in the text
./reviews/deep_22.md:9003:   750	texts in a batch of training samples. Therefore, the normalized
./reviews/deep_22.md:9014:   761	divergence of the image-to-text projection probability and the
./reviews/deep_22.md:9015:   762	true matching probability, the final image-to-text matching loss
./reviews/deep_22.md:9030:   777	After that, the image and text features in eq:15 also need
./reviews/deep_22.md:9031:   778	to be exchanged to calculate the text-to-image matching loss.
./reviews/deep_22.md:9151:   898	In order to accurately match the person image and text,
./reviews/deep_22.md:9156:   903	final person text features F D can be obtained. After that,
./reviews/deep_22.md:9176:/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | rg -n \"Ablation|TABLE II|Feature Distribution|Attention Visualization|PLM|MFGM|Lcpm|Lch|Ldcs|t-SNE|distance\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9247:/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | rg -n \"Ablation|TABLE|Distribution|Visualization|Parameter|containment|CMC|IMC|DRM|masked|uncertainty|Domain Generalization|Robustness\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9252:20:    20	Representation Modeling (DRM) encodes images and text into
./reviews/deep_22.md:9254:23:    23	distributions of text and masked text with their associated image
./reviews/deep_22.md:9260:113:   113	which aligns visual and textual distributions, and probabilistic containment, which imposes containment constraints
./reviews/deep_22.md:9262:116:   116	(DRM) module that encodes pedestrian images and texts into
./reviews/deep_22.md:9279:171:   171	For instance, IRRA [5] leverages masked language modeling to introduce implicit semantic reasoning, while CFine
./reviews/deep_22.md:9286:216:   216	incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
./reviews/deep_22.md:9290:235:   235	way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
./reviews/deep_22.md:9294:254:   254	learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
./reviews/deep_22.md:9306:392:   392	texts with low uncertainty is primarily determined by the
./reviews/deep_22.md:9307:394:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./reviews/deep_22.md:9308:395:   395	LDRM to align image-text pairs:
./reviews/deep_22.md:9314:422:   422	noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
./reviews/deep_22.md:9316:471:   471	between the sharpened image distribution and the text distribution. Formally, our containment objective acts as a
./reviews/deep_22.md:9319:622:   622	masked text feature distribution. By combining both complete
./reviews/deep_22.md:9320:623:   623	and masked text distributions, CMC encourages the image
./reviews/deep_22.md:9321:630:   630	the degree of containment rather than enforcing strict inclusion. This design ensures stable gradients even when textual
./reviews/deep_22.md:9340:707:   707	distribution Zvi with the masked text distribution Zmask
./reviews/deep_22.md:9341:710:   710	introduces uncertainty by randomly masking textual tokens.
./reviews/deep_22.md:9342:715:   715	or masked text due to the reduced visual information and
./reviews/deep_22.md:9354:750:   750	of the text than the image and LIMC enhance the model’s
./reviews/deep_22.md:9369:962:   962	the effects of complete and masked textual distributions. As
./reviews/deep_22.md:9370:968:   968	broadening the textual distribution’s coverage. The CMC and
./reviews/deep_22.md:9379:999:   999	A BLATION S TUDY ON RSTPR EID , E VALUATING THE I NDIVIDUAL AND C OMBINED E FFECTS OF DRM, CMC, AND IMC M ODULES ,
./reviews/deep_22.md:9402:1106:  1106	Fig. 4. Parameter sensitivity analysis of c on RSTPReid. The figure presents
./reviews/deep_22.md:9408:1156:  1156	proves that modeling visual-textual uncertainty through probabilistic distributions is a more efficient and powerful paradigm
./reviews/deep_22.md:9414:1222:  1222	the semantic generalization achieved through the CrossModal Containment (CMC) module, which models the textual
./reviews/deep_22.md:9424:1312:  1312	uncertainty modeling and semantic alignment for text-to-image person
./reviews/deep_22.md:9425:1464:  1464	uncertainty-based learning for text-based person retrieval,” in Proc.
./reviews/deep_22.md:9436:/bin/zsh -lc "pdftotext 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' - | nl -ba | rg -n \"TABLE|Fig\\. 3|Fig\\. 4|Optimization Strategy Comparison|Feature Compression Analysis|Tailed Semantics|joint optimization|Opt S|Opt JA|Opt JH|Enhanced\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9534:/bin/zsh -lc "pdftotext 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | rg -n \"Ablation|TABLE I|TABLE II|TABLE III|TABLE IV|TABLE V|visual|relationship|Acc|C2FL|Hc-Tri|CMRM|IMRM\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9572:998:   998	of text local feature relationships on the TI-ReID performance.
./reviews/deep_22.md:9573:999:   999	An analysis of the text contextual relationship-mining graph
./reviews/deep_22.md:9633:1324:  1324	[15] S. Li, T. Xiao, H. Li, W. Yang, and X. Wang, “Identity-aware textualvisual matching with latent co-attention,” in Proc. IEEE Int. Conf.
./reviews/deep_22.md:9634:1395:  1395	relationships: Attribute correlations in text attribute person search,” IEEE
./reviews/deep_22.md:9654:/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '910,990p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9672:   926	methods that aim to enhance the CLIP backbone, including IRRA, IRLT, and RDE, PDA demonstrates even more
./reviews/deep_22.md:9673:   927	significant improvements. Specifically, it surpasses RDE, the
./reviews/deep_22.md:9682:   936	higher than the noise-reduction-focused RDE [10], PDA
./reviews/deep_22.md:9685:   939	in training time per step compared to RDE, PDA achieves a
./reviews/deep_22.md:9687:   941	Furthermore, the computational demands of PDA are substantially lower than large-scale methods such as APTM [29]
./reviews/deep_22.md:9694:   948	offering a highly practical solution for large-scale text-based
./reviews/deep_22.md:9698:   952	contributions of each component in PDA, we conduct an ablation study on CUHK-PEDES, ICFG-PEDES, and RSTPReid,
./reviews/deep_22.md:9702:   956	confirms that representing pedestrian images and texts as
./reviews/deep_22.md:9708:   962	the effects of complete and masked textual distributions. As
./reviews/deep_22.md:9709:   963	shown in Rows 2 and 3, while both components individually surpass the baseline, the complete text Zt j provides a
./reviews/deep_22.md:9714:   968	broadening the textual distribution’s coverage. The CMC and
./reviews/deep_22.md:9724:   978	and 94.70%, respectively. Consistent improvements on ICFGPEDES and RSTPReid further confirm the complementary
./reviews/deep_22.md:9727:   981	2) Analysis of Baseline: We build our baseline by equipping IRRA [5] with standard training strategies from TBPR
./reviews/deep_22.md:9729:   983	in CFine [25], which is also used in RDE [10]. Although
./reviews/deep_22.md:9739:/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1180,1230p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9742:  1181	several text–image pairs. As shown, the Baseline model tends
./reviews/deep_22.md:9744:  1183	often ignoring fine-grained visual cues such as clothing texture, color patterns, and accessories. In contrast, PDA produces
./reviews/deep_22.md:9747:  1186	textual descriptions, such as “blue and black striped shirt”,
./reviews/deep_22.md:9749:  1188	that PDA is capable of capturing fine-grained visual–textual
./reviews/deep_22.md:9759:  1198	visually and textually similar identities. This observation confirms that the probabilistic embeddings not only enhance
./reviews/deep_22.md:9770:  1209	visual and textual representations of the top-10 retrieved results, with each
./reviews/deep_22.md:9772:  1211	represents the textual query with a 90% confidence region.
./reviews/deep_22.md:9778:  1217	images and their corresponding textual queries. As shown
./reviews/deep_22.md:9780:  1219	textual query embedding, while the colored points represent
./reviews/deep_22.md:9782:  1221	The broader spatial coverage of the textual ellipses reflects
./reviews/deep_22.md:9783:  1222	the semantic generalization achieved through the CrossModal Containment (CMC) module, which models the textual
./reviews/deep_22.md:9785:  1224	deterministic point. This design enables the text distribution
./reviews/deep_22.md:9794:/bin/zsh -lc "pdftotext 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '1178,1238p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9796:/bin/zsh -lc "pdftotext 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '1048,1120p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:9798:  1048	person image and text features respectively. In the experiment, the extracted features are directly used for similarity
./reviews/deep_22.md:9799:  1049	calculation to match person images and texts. Table IV shows
./reviews/deep_22.md:9884:  1189	Similarly, compared with the second-ranked method APTM
./reviews/deep_22.md:9892:  1197	methods on the RSTPReid dataset. The experiment results are
./reviews/deep_22.md:9902:  1207	some person text descriptions from the CUHK-PEDES dataset
./reviews/deep_22.md:9908:  1213	text and the image, enabling accurate cross-modal person re-
./reviews/deep_22.md:9916:  1221	Fig. 6. Visualization of the text-to-image person re-identification comparison results on the CUHK-PEDES dataset. The first row is the retrieval results of
./reviews/deep_22.md:9954:/bin/zsh -lc "pdftotext 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' - | nl -ba | sed -n '359,470p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10070:/bin/zsh -lc "pdftotext 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' - | nl -ba | sed -n '468,690p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10297:/bin/zsh -lc "pdftotext 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | sed -n '1390,1448p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10299:/bin/zsh -lc "pdftotext 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | sed -n '1260,1405p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10522:/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | sed -n '540,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10611:/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | sed -n '777,815p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10649:   813	context of cross-modality person re-identification.
./reviews/deep_22.md:10654:/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | sed -n '792,815p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10677:   813	context of cross-modality person re-identification.
./reviews/deep_22.md:10682:/bin/zsh -lc "pdftotext 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '933,1247p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:10691:   940	Effectiveness of CSIA. CSIA constrains the consistent similarity between intra-modality image-text pairs and intermodality text-text pairs to establish the correspondence between images and cross-modality texts. As detailed in Table
./reviews/deep_22.md:10987:  1236	ment of inter-modality image-text pairs plays a positive role
./reviews/deep_22.md:10989:  1238	Effectiveness of CVSC. CVSC integrates multi-view texts to
./reviews/deep_22.md:11001:PDM 的表格在普通文本抽取里排版有些乱，但它的文字说明和图分析已经足够支撑方法论判断：组件消融证明 PLM、MFGM、两个约束项都贡献；距离分布和 t-SNE 证明它真的扩大跨身份间隔、缩小同身份跨模态间隔。
./reviews/deep_22.md:11019:/bin/zsh -lc "pdftotext -layout -f 1 -l 8 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | sed -n '270,350p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:11104:/bin/zsh -lc "pdftotext -layout 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | sed -n '650,710p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:11169:/bin/zsh -lc "pdftotext -layout 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | sed -n '760,840p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:11179:H. Qualitative Results                                                           textual descriptions, such as “blue and black striped shirt”,
./reviews/deep_22.md:11181:the effectiveness of our method, Fig. 5 illustrates the                          that PDA is capable of capturing fine-grained visual–textual
./reviews/deep_22.md:11190:                                                                                visually and textually similar identities. This observation con-
./reviews/deep_22.md:11206:                                                                                embeddings, PDA models images and text as Gaussian
./reviews/deep_22.md:11209:                                                                                to explore the broader semantic scope of text compared
./reviews/deep_22.md:11210:                                                                                to image, ensuring that text distributions encompass paired
./reviews/deep_22.md:11223:visual and textual representations of the top-10 retrieved results, with each
./reviews/deep_22.md:11225:represents the textual query with a 90% confidence region.                      a promising paradigm for improving both robustness and
./reviews/deep_22.md:11236:images and their corresponding textual queries. As shown                        [3]   S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang, “TransReID:
./reviews/deep_22.md:11239:textual query embedding, while the colored points represent                     [4]   S. Li, S. Li, and Q. Li, “CLIP-ReID: Exploiting vision-language model
./reviews/deep_22.md:11240:the top-10 retrieved image embeddings from the gallery.                               for image re-identification without concrete text labels,” in Proc. AAAI,
./reviews/deep_22.md:11241:The broader spatial coverage of the textual ellipses reflects                         2023, vol. 37, no. 1, pp. 1405–1413.
./reviews/deep_22.md:11243:the semantic generalization achieved through the Cross-                               aligning for text-to-image person retrieval,” in Proc. IEEE/CVF Conf.
./reviews/deep_22.md:11244:Modal Containment (CMC) module, which models the textual                              Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 2787–2797.
./reviews/deep_22.md:11245:representation as a probabilistic distribution rather than a                    [6]   T. Fujii and S. Tarashima, “BiLMa: Bidirectional local-matching for text-
./reviews/deep_22.md:11247:deterministic point. This design enables the text distribution                        Vis. Workshops (ICCVW), Oct. 2023, pp. 2778–2782.
./reviews/deep_22.md:11249:(e.g., ID 3955), while excluding semantically irrelevant or neg-                      for text-based person retrieval,” in Proc. Int. Conf. Multimedia Retr.,
./reviews/deep_22.md:11254:/bin/zsh -lc "pdftotext -layout 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | sed -n '1030,1095p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_22.md:11261:1. **触发观察**：作者抓住了文本行人检索里的非对称性，文本通常是粗粒度描述，图像是具体实例。一个文本本来对应一组可能图像，而不是一个点对一个点。现有点嵌入把强正样本、弱正样本、相似负样本都挤在同一套距离逻辑里，容易把“语义范围”误当成“特征距离”。
./reviews/deep_22.md:11263:2. **重定义动作**：它把问题从“图文相似度对齐”改写成“文本分布应当包含图像分布”。关键词是 probabilistic containment、semantic scope、partial-order relationship、text as probabilistic envelope。厉害的地方是把方差从“噪声余量”重新定义成“语义覆盖范围”。
./reviews/deep_22.md:11265:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reviews/deep_22.md:11267:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./reviews/deep_22.md:11305:5. **reviewer 为什么买账**：这篇卖的是反常识视角和干净证据，不是新模块。它用 ReID 构造了一个能隔离混杂因素的实验场景，证明“联合优化好”这个共识并不总成立。方法论价值是：如果能拆掉领域里的混淆变量，负结论也能变成贡献。
./reviews/deep_22.md:11311:2. **重定义动作**：它把 VI-ReID 从“视觉特征对齐到公共空间”改成“视觉特征对齐到显式且丰富的语义空间”。关键词是 explicit semantics、enriched semantics、indirect alignment、cross-view semantics compensation、one-to-many alignment。它不是简单说“用文本监督”，而是拆成显式语义、跨模态语义冲突、多视角语义不完整三个子问题。
./reviews/deep_22.md:11313:3. **机制怎么长出来**：ESFA 用 LLaVA 给可见光和红外图像生成描述，在各自模态内做图文对齐。CSIA 不直接强拉图像和异模态文本，而是约束“模态内图文相似度”和“跨模态文本相似度”一致，绕开颜色冲突。CVSC 用同身份其他视角文本补偿当前文本，把一对一图文匹配扩成一对多语义对齐。机制和重定义绑定非常紧。
./reviews/deep_22.md:11315:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./reviews/deep_22.md:11321:1. **触发观察**：TI-ReID 里很多方法已经做局部图文对齐，但作者指出只看局部对象还不够。两个人都可能有灰外套、黑包，差别在“背着包”还是“穿着包”这样的局部关系。另一个观察是 Hc-Tri loss 很早收敛到 0，后期没有有效监督。
./reviews/deep_22.md:11325:3. **机制怎么长出来**：IMRM 在图像局部和文本局部内部建图，用 GNN 聚合关系信息。TCRMG 特别做了正序和倒序文本图，因为倒序能更早看到“carrying bag”和“wearing bag”这种区分短语。CMRM 再用跨模态 KNN 图建图，学习图文局部之间的关系，并做二分类判断是否匹配。C2FL 先用较松的 Hc-Tri，等它收敛后逐步切到更严格的 Acc loss。
./reviews/deep_22.md:11338:1. **触发观察**：作者抓住了文本行人检索里的非对称性，文本通常是粗粒度描述，图像是具体实例。一个文本本来对应一组可能图像，而不是一个点对一个点。现有点嵌入把强正样本、弱正样本、相似负样本都挤在同一套距离逻辑里，容易把“语义范围”误当成“特征距离”。
./reviews/deep_22.md:11340:2. **重定义动作**：它把问题从“图文相似度对齐”改写成“文本分布应当包含图像分布”。关键词是 probabilistic containment、semantic scope、partial-order relationship、text as probabilistic envelope。厉害的地方是把方差从“噪声余量”重新定义成“语义覆盖范围”。
./reviews/deep_22.md:11342:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./reviews/deep_22.md:11382:5. **reviewer 为什么买账**：这篇卖的是反常识视角和干净证据，不是新模块。它用 ReID 构造了一个能隔离混杂因素的实验场景，证明“联合优化好”这个共识并不总成立。方法论价值是：如果能拆掉领域里的混淆变量，负结论也能变成贡献。
./reviews/deep_22.md:11388:2. **重定义动作**：它把 VI-ReID 从“视觉特征对齐到公共空间”改成“视觉特征对齐到显式且丰富的语义空间”。关键词是 explicit semantics、enriched semantics、indirect alignment、cross-view semantics compensation、one-to-many alignment。它不是简单说“用文本监督”，而是拆成显式语义、跨模态语义冲突、多视角语义不完整三个子问题。
./reviews/deep_22.md:11390:3. **机制怎么长出来**：ESFA 用 LLaVA 给可见光和红外图像生成描述，在各自模态内做图文对齐。CSIA 不直接强拉图像和异模态文本，而是约束“模态内图文相似度”和“跨模态文本相似度”一致，绕开颜色冲突。CVSC 用同身份其他视角文本补偿当前文本，把一对一图文匹配扩成一对多语义对齐。机制和重定义绑定非常紧。
./reviews/deep_22.md:11392:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./reviews/deep_22.md:11398:1. **触发观察**：TI-ReID 里很多方法已经做局部图文对齐，但作者指出只看局部对象还不够。两个人都可能有灰外套、黑包，差别在“背着包”还是“穿着包”这样的局部关系。另一个观察是 Hc-Tri loss 很早收敛到 0，后期没有有效监督。
./reviews/deep_22.md:11402:3. **机制怎么长出来**：IMRM 在图像局部和文本局部内部建图，用 GNN 聚合关系信息。TCRMG 特别做了正序和倒序文本图，因为倒序能更早看到“carrying bag”和“wearing bag”这种区分短语。CMRM 再用跨模态 KNN 图建图，学习图文局部之间的关系，并做二分类判断是否匹配。C2FL 先用较松的 Hc-Tri，等它收敛后逐步切到更严格的 Acc loss。
./OSAC_KILLSWITCH_DESIGN.md:1:# OSAC (Open-Set Spectral Over-Collapse) kill-switch 设计
./OSAC_KILLSWITCH_DESIGN.md:12:- vs hubness(HAL/NeighborRetr): hubness 只当 readout symptom, remedy 是 training-spectral 不是 retrieval-hub-fix。
./OSAC_KILLSWITCH_DESIGN.md:14:## ★零训练 kill-switch（接近 0-GPU, 先只分析现有 checkpoint, 不训练）
./OSAC_KILLSWITCH_DESIGN.md:21:- gallery hubness H_k + query hub mass M(q)(复用 hubness 脚本)。
./OSAC_KILLSWITCH_DESIGN.md:25:2. **坍缩↔检索失败相关**: per-query AP error ~ (query 在 top-PC 上的投影能量 / prototype alignment) 相关, **控 camera/norm/margin 后仍显著**。hub mass M(q) ~ top-PC energy 相关(证 hubness 是坍缩症状)。
./OSAC_KILLSWITCH_DESIGN.md:26:3. **去坍缩诊断干预(零训练)**: 对 embedding 做 ABTT(去 top-m PC) / whitening, 看 (a)hubness 是否降, (b)raw mAP 是否可见提升。
./OSAC_KILLSWITCH_DESIGN.md:34:**通过标准:** 过坍缩轨迹明确(ep后期 rank↓) + AP error ~ top-PC energy partial 显著 + ABTT 降 hubness 涨 raw + **D1 随机PC不如top-PC + D2 ABTT 在 k-reciprocal 之后仍有残余增益 + D3 控代理后仍在**。
./OSAC_KILLSWITCH_DESIGN.md:35:→ 全过(尤其 D2 残余) = 过坍缩是 k-reciprocal 拿不到的真 training-side 失败变量 → 单训练 kill-switch(强 baseline 加 OSAC: spectral floor + Top-PC Dropout, 测试仍单 embedding; ep30/60 看 effective rank↑/top-PC hub correlation↓/raw +0.5 稳; 成功线 raw +0.8~1.0 且 k-reciprocal/camera 后仍 +0.3 残余)。
./OSAC_KILLSWITCH_DESIGN.md:38:## 机制草案(过了 kill-switch 才做)
./OSAC_KILLSWITCH_DESIGN.md:42:多 epoch ckpt 在 lab-3090 log/。复用 cvpb_hubness/gopl kill-switch 的 extract/per_query_ap/H_k 基建。谱分析纯 numpy。
./debate/clean/d_2.txt:1:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./debate/clean/d_2.txt:3:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./debate/clean/d_2.txt:7:1. **hubness-aware training 不是空白，但证据不迁移到 ReID。**  
./debate/clean/d_2.txt:8:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./debate/clean/d_2.txt:16:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./debate/clean/d_2.txt:35:- 只有在额外零训练 kill-switch 上同时满足这些门槛才重新升级：OccludedDuke 或 MSMT 上 mAP 增益显著高于 k-reciprocal/camera；hub penalty 与 k-reciprocal 组合后仍有独立 mAP 增益；不是只涨 R1。
./debate/clean/d_2.txt:38:信心：**8/10**。不是说 hubness 是错的，而是说“诊断真”到“训练版可发”中间缺的正是 D2 没过的那一关。
./analyses/deep_9.txt:11:1. 触发观察：CLIP-ReID 的 learnable prompt 有判别性，但缺少具体人的描述先验。MLLM 生成的描述有人的属性细节，但文本会有噪声。已有多模态 ReID 把 learnable text 和 description text 同步融合，等价于把可靠和不可靠信息同等对待。
./analyses/deep_9.txt:12:2. 重定义动作：它把问题从“文本能不能帮助 ReID”改成“两类文本嵌入的可靠性和粒度不同，不能对称使用”。关键词是 learnable-based textual embedding、description-based textual embedding、asymmetric fusion、auxiliary clues、cross-granularity training。
./analyses/deep_9.txt:13:3. 机制怎么长出来：因为 description text 有信息但不够可靠，所以它不让描述文本当决定性监督，而是作为辅助线索注入 visual embedding 和 learnable textual embedding。Textual-to-Visual Adapter 用描述去补视觉，Textual-to-Textual Adapter 用描述去补身份 prompt，再用图像级描述和身份级文本做交叉粒度训练。
./analyses/deep_9.txt:14:4. 证据闭环：核心消融是融合方式和文本约束。cross-attention 比直接相加好很多，直接相加甚至会明显伤性能，说明“非对称融合”不是装饰。损失消融里，只有 learnable text、只有 description text、二者组合和完整约束逐步提升，说明描述文本确实是辅助增强而不是替代视觉或替代 prompt。
./analyses/deep_9.txt:15:5. reviewer 为什么买账：买的是“文本空间不等价”的融合视角。真正 novelty 不是 adapter，而是把 MLLM 描述的噪声风险讲清楚，然后给出不让噪声文本主导训练的非对称设计。不过这篇的增益相对克制，更像 ICME 级别的稳健增量，不是特别强的问题重定义范式。
./analyses/deep_9.txt:25:1. 触发观察：无监督 text-to-image ReID 不只是没有身份标签，还没有可靠的图文匹配关系。图像聚类能给伪标签，但文本不一定和图像匹配，MLLM 生成文本也会有幻觉和冗余属性。
./analyses/deep_9.txt:26:2. 重定义动作：它把旧问题“无监督跨模态学习”改成“训练样本和伪标签的 dependability 问题”。关键词是 dependable text generation、reliable image-text matching relationship、Error Sample Filtering、dependable identity labels。
./analyses/deep_9.txt:27:3. 机制怎么长出来：既然核心是可靠性，它先用多个 MLLM 生成候选描述，再用已有 TIReID 模型的图文相似度选择更可靠文本。然后只对图像聚类，把伪标签传给匹配文本，再用文本和图像类中心的相似度做 IQR 异常过滤。最后 MTJL 从类中心和实例两个层面收紧跨模态特征。
./analyses/deep_9.txt:28:4. 证据闭环：消融是比较完整的。单 MLLM 生成文本加聚类最差，加入 MTJL 有明显提升，换 MLLM 会波动，说明文本质量不稳定。dependable text generation 明显提升，ESFM 再提升。IQR 和 Z-score、LOF 对比，聚类方法替换，过滤强度参数分析，都在证明“可靠样本构造”是承重点。
./analyses/deep_9.txt:29:5. reviewer 为什么买账：买的是问题分解。它没有简单说“大模型生成文本能用”，而是说无监督 TIReID 的难点在两层可靠性：图文关系可靠、伪身份可靠。novelty 来源是数据构造和可靠性过滤的闭环，机制和重定义绑定较紧。
./analyses/deep_9.txt:45:这批论文共同套路是：先不急着加模块，而是把旧范式的副作用单独命名，比如对齐损判别性、文本有噪声、无监督样本不可靠、序列级仍残留模态信息、单频切分会丢身份线索。然后造一个中间概念承接新问题，比如 neutral feature、dependable pair、ID-related sequence representation、multi-frequency embedding，再让模块围绕这个中间概念服务。真正能让 reviewer 买账的不是模块复杂，而是消融和可视化能证明“这个新问题定义本身是对的”。
./hubness_debate.sh:8:CTX="一个做行人 ReID 的团队要发 CCF-B 方法稿。连续两个方向被零训练 kill-switch 干净杀死后(航拍-不确定性包含/SMPL-正样本可靠性, 都是错的隐藏变量), 第三个方向 **Gallery Hubness** 的零训练 kill-switch 出了 nuanced 结果, 现在要决定**该不该投入第一次训练**去做训练版。
./hubness_debate.sh:12:**零训练 kill-switch 结果(冻结 Market 强 ckpt exp260b, sanity cosine mAP94.61≈训练94.4)**:
./hubness_debate.sh:13:- 诊断关全过: ★D4 决定性——负向 in-degree 与'热门样本'**正交**(rho(AP误差,M_neg)=+0.28 但 M_all=−0.08 符号翻转, Spearman(H_neg,H_all)=−0.029 全量近无关)。hub mass 解释 AP 误差完胜 norm/margin/camera/#pos(partial 控住全部仍+0.33)。top1% hub 吃 22-53% false-top1。D1 置换破(增益消失)。novelty-check: ReID 里无确切先例, 但跨模态检索 HAL(CVPR20)/NeighborRetr(CVPR25) 已做训练端 hubness-aware loss+memory bank(任务是图文非 person ReID)。
./hubness_debate.sh:19:ROLES[1]="角色=**红队(投训练 anti-hub)**。为'投入第一次训练做 anti-hub embedding'辩护: 诊断真+新(D4 干净, ReID 无先例), hub 在 R1 互补不是被压, Market 只是天花板太低(换 occluded_duke/MSMT 未饱和 headroom 大)。用联网论证: (a)训练端 anti-hub embedding 能给比 test-time k-reciprocal 更好的**单向量**(且二者互补, 实践常并用); (b)切口避开 HAL/NeighborRetr(它们图文/跨模态, 我们 person ReID gallery 拓扑); (c)在更难 benchmark hub 效应更大。给最小训练验证方案(单数据单训练即可判) + 信心 1-10。"
./hubness_debate.sh:20:ROLES[2]="角色=**蓝队(降级 Hubness)**。为'别投训练, 降级转 r_2 备胎 Rank-Instability'辩护: D2 是铁证——k-reciprocal/camera **免费 test-time** 就在 mAP 上赢, 一个训练版要 beat camera-aware k-reciprocal 是高 bar 且小 headroom(Market 饱和)。用联网查: (a)hubness-aware training / anti-hub margin 历史上是否真能 beat 强 re-ranking(还是总被 k-reciprocal 这类盖过); (b)R1 赢 mAP 输是不是只是把 ranking 重排没真增加判别信息; (c)团队反复投不确定训练的教训。论证'诊断真≠方法能发', 该把诊断当一个 observation 写进别的稿/换 r_2。信心 1-10。"
./debate/clean/d_1.txt:1:**结论：该投，但只能投一次“小训练判生死”。**  
./debate/clean/d_1.txt:4:文献侧支持“训练值得一试”。Hubness 本来就是高维 kNN/检索空间里的结构性问题，HAL 在 text-image matching 中明确用 local/global hub 统计和 memory bank 做训练端 hubness-aware loss，并报告 R@1 大幅提升；这说明“训练时改 embedding 分布”比事后惩罚分数有现实先例。([arxiv.org](https://arxiv.org/abs/1911.10097)) NeighborRetr 也把 cross-modal retrieval 的 hubness 从 post-hoc normalization 推到 training-time balancing，并强调 bad hubs / good hubs 的区分。([github.com](https://github.com/zzezze/NeighborRetr))
./debate/clean/d_1.txt:6:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./debate/clean/d_1.txt:32:不是 7/10 保证成稿，而是 7/10 认为“这一次训练的信息价值大于成本”。若 Occluded-Duke 单训练同时提升单向量指标并压低 hub mass，这条线就值得进入正式实验；若不行，应直接 kill，不在 Market 小修补。
./analyses/deep_10.txt:39:3. 机制怎么长出来：全局表征负责整个人，属性表征负责细粒度局部属性。ATG 先用属性识别器生成属性文本，TGA 用 CLIP 融合图文得到全局表征，ACN 得到属性级表征；AF 用属性表征对齐旧新模型，KC 用全局表征做知识转移。
./analyses/deep_10.txt:40:4. 证据闭环：它用跨五个数据集的 t-SNE 证明域间一致性，用 Market 的身份聚类可视化证明域内判别增强。组件消融显示 PFM、ACN、AF、KC 分别贡献；ATG 用具体属性文本优于泛化的 “a photo of a person”；阈值 0.8 的消融说明属性可靠性被认真控制。
./batches/bag:4:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./debate/d_1.md:14:一个做行人 ReID 的团队要发 CCF-B 方法稿。连续两个方向被零训练 kill-switch 干净杀死后(航拍-不确定性包含/SMPL-正样本可靠性, 都是错的隐藏变量), 第三个方向 **Gallery Hubness** 的零训练 kill-switch 出了 nuanced 结果, 现在要决定**该不该投入第一次训练**去做训练版。
./debate/d_1.md:18:**零训练 kill-switch 结果(冻结 Market 强 ckpt exp260b, sanity cosine mAP94.61≈训练94.4)**:
./debate/d_1.md:19:- 诊断关全过: ★D4 决定性——负向 in-degree 与'热门样本'**正交**(rho(AP误差,M_neg)=+0.28 但 M_all=−0.08 符号翻转, Spearman(H_neg,H_all)=−0.029 全量近无关)。hub mass 解释 AP 误差完胜 norm/margin/camera/#pos(partial 控住全部仍+0.33)。top1% hub 吃 22-53% false-top1。D1 置换破(增益消失)。novelty-check: ReID 里无确切先例, 但跨模态检索 HAL(CVPR20)/NeighborRetr(CVPR25) 已做训练端 hubness-aware loss+memory bank(任务是图文非 person ReID)。
./debate/d_1.md:25:角色=**红队(投训练 anti-hub)**。为'投入第一次训练做 anti-hub embedding'辩护: 诊断真+新(D4 干净, ReID 无先例), hub 在 R1 互补不是被压, Market 只是天花板太低(换 occluded_duke/MSMT 未饱和 headroom 大)。用联网论证: (a)训练端 anti-hub embedding 能给比 test-time k-reciprocal 更好的**单向量**(且二者互补, 实践常并用); (b)切口避开 HAL/NeighborRetr(它们图文/跨模态, 我们 person ReID gallery 拓扑); (c)在更难 benchmark hub 效应更大。给最小训练验证方案(单数据单训练即可判) + 信心 1-10。
./debate/d_1.md:35:web search: 'directly mitigate hubness during training' in https://arxiv.org/abs/2503.10526
./debate/d_1.md:37:web search: person re-identification hubness gallery hubness hub-aware loss
./debate/d_1.md:39:web search: "gallery" "hubness" "re-identification"
./debate/d_1.md:41:web search: hubness high-dimensional spaces k-nearest neighbor skewness Radovanovic 2010
./debate/d_1.md:73:web search: github HAL Improved Text-Image Matching Mitigating Visual Semantic Hubs hubness-aware loss
./debate/d_1.md:79:**结论：该投，但只能投一次“小训练判生死”。**  
./debate/d_1.md:82:文献侧支持“训练值得一试”。Hubness 本来就是高维 kNN/检索空间里的结构性问题，HAL 在 text-image matching 中明确用 local/global hub 统计和 memory bank 做训练端 hubness-aware loss，并报告 R@1 大幅提升；这说明“训练时改 embedding 分布”比事后惩罚分数有现实先例。([arxiv.org](https://arxiv.org/abs/1911.10097)) NeighborRetr 也把 cross-modal retrieval 的 hubness 从 post-hoc normalization 推到 training-time balancing，并强调 bad hubs / good hubs 的区分。([github.com](https://github.com/zzezze/NeighborRetr))
./debate/d_1.md:84:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./debate/d_1.md:110:不是 7/10 保证成稿，而是 7/10 认为“这一次训练的信息价值大于成本”。若 Occluded-Duke 单训练同时提升单向量指标并压低 hub mass，这条线就值得进入正式实验；若不行，应直接 kill，不在 Market 小修补。
./debate/d_1.md:113:**结论：该投，但只能投一次“小训练判生死”。**  
./debate/d_1.md:116:文献侧支持“训练值得一试”。Hubness 本来就是高维 kNN/检索空间里的结构性问题，HAL 在 text-image matching 中明确用 local/global hub 统计和 memory bank 做训练端 hubness-aware loss，并报告 R@1 大幅提升；这说明“训练时改 embedding 分布”比事后惩罚分数有现实先例。([arxiv.org](https://arxiv.org/abs/1911.10097)) NeighborRetr 也把 cross-modal retrieval 的 hubness 从 post-hoc normalization 推到 training-time balancing，并强调 bad hubs / good hubs 的区分。([github.com](https://github.com/zzezze/NeighborRetr))
./debate/d_1.md:118:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./debate/d_1.md:144:不是 7/10 保证成稿，而是 7/10 认为“这一次训练的信息价值大于成本”。若 Occluded-Duke 单训练同时提升单向量指标并压低 hub mass，这条线就值得进入正式实验；若不行，应直接 kill，不在 Market 小修补。
./analyses/deep_19.txt:17:1. 触发观察：已有换衣 ReID 往往回避衣服信息，或者用非衣服描述如体型、姿态、轮廓来学习不变特征。但这些非衣服描述经常不精确，CLIP 文本和细粒度图像之间又有语义差距。作者反过来观察到，衣服描述虽然不该成为最终身份特征，但它可以告诉模型“哪些东西应该被压掉”。
./analyses/deep_19.txt:19:2. 重定义动作：旧问题是“如何避免模型依赖衣服”；新问题是“如何准确定位衣服语义并把它从视觉特征中减掉”。这一步很关键，因为作者不是说衣服文本有用来识别人，而是说衣服文本有用来抑制衣服。关键词是 clothing descriptions、coarse-to-fine、semantic-guided、clothes-agnostic、parsing image。
./analyses/deep_19.txt:21:3. 机制怎么长出来：机制和重定义绑定较紧。先用 human parsing 抠出衣服区域，让多个 LVLM 只描述衣服，并用粗粒度和细粒度提示生成多种衣服 caption。再用 CLIP 相似度过滤，保留最可信的两个描述。SGE 分支让图像特征和衣服文本做注意力，然后从视觉特征里减去衣服相关成分。PIE 分支直接遮掉衣服区域训练，让模型被迫使用更稳定的身体、姿态、局部线索。
./analyses/deep_19.txt:41:1. 触发观察：作者看到现有多模态换衣 ReID 没把属性用充分。AD-ViT、MADE 一类用属性标签，但多是一维 one-hot；CCAF、MIPL 一类用文本，但缺少结构化属性信息；很多方法还依赖人工衣服标签。另一个观察是，像素级遮衣服能做低层解耦，但缺少高层语义指导。
./analyses/deep_19.txt:43:2. 重定义动作：旧问题是“属性能不能作为辅助标签”；新问题是“属性文本能不能作为多粒度提示，明确规定哪些语义该保留、哪些该压制”。关键词是 multi-granularity attribute prompt、attribute-level、caption-level、identity-related、clothing-related、pseudo label。
./analyses/deep_19.txt:45:3. 机制怎么长出来：机制很顺。先用 SCHP 生成衣服擦除图，再用 AAPAR 识别 35 类属性。年龄、性别被当作身份相关属性，头饰、上衣、下装、鞋、配饰等被当作衣服相关属性。然后用模板把属性变成文本，在 raw image branch 和 masked image branch 上用 CLIP ViT 加 visual deep prompt 学图像表示。属性解耦部分把图像特征和身份相关文本拼接，用 triplet 促进保留；对衣服相关文本用负向损失或对抗约束进行压制。BNNeck 用来隔开图像调优和文本对齐，避免一个目标破坏另一个目标。
./analyses/deep_19.txt:47:4. 证据闭环：消融很对题。baseline 加 clothing adversarial attribute 后提升，说明压制衣服属性有效；加 identity-relevant attribute 后继续提升，说明保留稳定属性有效；再加 caption-level 约束提升，说明多粒度文本不是摆设；masked branch 的属性解耦去掉后也下降，说明擦衣服后仍有残留衣服语义。检索可视化显示 baseline 受颜色和款式影响更重，MAPL 更能跨衣服找人。
./analyses/deep_19.txt:49:5. 审稿人为什么买账：这篇卖的是视角。它不是“又用了属性”，而是把属性拆成正向身份语义和负向衣服语义，再用文本提示做多粒度解耦。真正新意来自“属性从标签变成 prompt 化的解耦约束”，同时减少人工衣服标签依赖。
./analyses/deep_28.txt:6:作者注意到一个反直觉点：在文本行人检索里，全量微调 CLIP 不一定是最好的。全量微调虽然能适配训练集，但会破坏或遗忘 CLIP 原本的大规模视觉语言知识，而且每个数据集都要存一套大模型。相反，参数高效微调保留 CLIP 主干，可能同时利用“CLIP 里已有的行人相关知识”和“训练集学到的任务知识”。第二个观察更具体：把 Prefix、LoRA、Adapter 直接堆在一起会互相干扰，普通 Prefix 在 TPR 上收敛慢、效果差。
./analyses/deep_28.txt:15:关键证据不是单纯 SOTA 表，而是组件消融和冲突消融。单独的 LoRA、S-Prefix、L-Adapter 都比 zero-shot CLIP 强，三者组合在 CUHK-PEDES 上达到最高 R@1 和 mAP，说明“全局、局部、任务提示”确实互补。S-Prefix 对 vanilla Prefix 有明显提升，并且缩放因子过大时会梯度爆炸，证明他们不是随便加参数，而是在修 Prefix 梯度问题。L-Adapter 对比 vanilla Adapter、LN-tuning 和混合 Adapter，支持“普通 Adapter 会和其他 PETL 冲突，LayerNorm 旁边更适合调全局分布”的说法。可视化里 UP-Person 比全量微调的 IRRA 更能识别细粒度或少见描述，也服务于“全量微调丢失 CLIP 原知识”的叙事。
./analyses/deep_28.txt:40:作者抓住的是 CLIP-ReID 类方法里的一个弱点：两阶段生成的 text prompts 往往不关注行人身份细节。原因是原始 CLIP 对 VI-ReID 不熟，直接生成的提示更偏通用语义，不一定关心跨模态行人身份。另一个观察是，如果同时优化 prompts 和 image encoders，二者目标不同，会互相干扰。
./analyses/deep_28.txt:43:它把 prompt 从“给图像编码器用的静态文本辅助”改写成“可学习的中间模态”。关键词是 mutual promotion、mutual reinforcement、person-related prompts、text features as bridge、modality knowledge propagation、modality alignment、alternating training。旧问题是“怎么用文本帮助可见光和红外对齐”，新问题是“提示和图像编码器怎样互相变好”。
./analyses/deep_28.txt:46:机制很自然。既然 prompts 缺少身份信息，就让训练好的图像编码器把身份知识传给 prompts。作者用分类概率分布里的 dark knowledge，通过 KL 让文本特征的类别概率接近图像特征的类别概率。既然高质量 prompts 可以作为模态无关桥梁，就在优化图像编码器时，把可见光和红外图像特征都对齐到文本特征。既然同时优化会互相干扰，就交替训练：偶数 epoch 固定图像编码器训 prompts，奇数 epoch 固定 prompts 训图像编码器。这个设计和“互相强化”的叙事绑定很紧。
./analyses/deep_28.txt:49:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./analyses/deep_28.txt:52:这篇主要卖视角。模块上就是 KL 蒸馏、对齐损失、交替优化，但它把 prompt 的角色从“静态文本”翻成“被图像编码器教育、再反过来教育图像编码器的中间模态”。这比单纯说“引入文本提示”更像一个方法故事。reviewer 买账的原因是机制简单，问题清楚，证据能直接回答“prompt 是否真的更关注身份”。
./analyses/deep_28.txt:74:作者注意到 LVLM 很强，但直接用于 ReID 很别扭。让 LVLM 逐个判断 query 和 gallery 是否同人，成本和 token 数都不可接受。让 LVLM 生成 caption 再做 image-text matching 也有问题：高质量文本标注贵，caption 目标和 ReID 目标不一致，推理阶段通常没有真正利用 LVLM 的生成能力。CLIP-ReID 类方法虽然用了文本，但往往训练后丢掉文本编码器。
./analyses/deep_28.txt:77:它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。
./analyses/deep_28.txt:80:机制也比较自然。为了不逐对比对，就仍然保留 ReID 的特征提取加相似度检索范式。为了用 LVLM 的语义理解能力，就用指令让 Qwen2-VL 针对行人图像生成一个 `<REID>` 语义 token，要求关注年龄、性别、衣着、生物特征。为了让这个 token 不只是孤立文本输出，设计 SGI，把语义 token 和视觉 tokens 拼起来经过 4 层 Transformer 交互，最后用强化后的语义 token 作为身份表示。为了处理相机变化，又加 camera semantic supplementation，最好的是把相机 embedding 加到视觉 patch embedding 上。训练用 ID loss 和 triplet loss，LLM 冻结但允许梯度穿过它更新视觉模型和 SGI。
./reviews/deep_9.md:14:你是 ReID 创新方法论拆解员 9。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。
./reviews/deep_9.md:36:我会按项目规则先确认能读到的上下文，然后用 PDF 技能和 `pdftotext` 逐篇抽方法部分。当前沙箱是只读，我不会写入任何文件。
./reviews/deep_9.md:58:A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./reviews/deep_9.md:59:A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./reviews/deep_9.md:69:Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./reviews/deep_9.md:76:CLIP-driven fine-grained mining for text-based person search.pdf
./reviews/deep_9.md:89:ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./reviews/deep_9.md:93:Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./reviews/deep_9.md:137:Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./reviews/deep_9.md:144:Internal-External Context Interaction Network for Person Re-Identification.pdf
./reviews/deep_9.md:187:See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./reviews/deep_9.md:200:TP-LReID - Lifelong person re-identification using text prompts.pdf
./reviews/deep_9.md:224:/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_9.md:239:# PDF Skill
./reviews/deep_9.md:253:3. Use `pdfplumber` or `pypdf` for text extraction and quick checks; do not rely on text extraction for layout fidelity.
./reviews/deep_9.md:308:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reviews/deep_9.md:390:2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_9.md:403:第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。
./reviews/deep_9.md:430:两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。
./reviews/deep_9.md:437:- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。
./reviews/deep_9.md:510:第一，做训练前两轮独立审查里的 Codex 这一轮。一个改了方法的实验，在开始训练之前，主 agent 会用 `codex exec` 起你来做一次独立代码审查。你看不到另一个审查者的结论，也不知道这是第几轮，也不要假设自己知道改了什么。审查范围是设计文档、新增和改动代码、配置、对照组变量隔离、评测协议和数据流。
./reviews/deep_9.md:512:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reviews/deep_9.md:528:凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
./reviews/deep_9.md:546:自己写的评测脚本或分析脚本，要先用它复现一个已知的基线成绩。对得上之后，才能用它的结果下结论。新的评测口径、新的度量、新的脚本都按这条执行。
./reviews/deep_9.md:615:任何改了模型或者有新设计的实验，在启动训练之前，都要经过两轮互不通气的独立审查。一轮由 Claude 做，一轮由 Codex 做。两个审查者互相看不到对方结论，也不知道这是第几轮。
./reviews/deep_9.md:617:Codex 这一轮要完整审一遍，不是只看某几处。要看设计文档、新增和改动的代码、配置、对照组、变量隔离和评测协议。结论写进 `experiments/exp{编号}/review-codex.md`。
./reviews/deep_9.md:660:重大决策前，尽量让两个独立视角辩一辩。一个为方案 A 辩护，一个为方案 B 辩护，从技术可行性、创新性、论文价值、风险、成本几方面讲，并给出信心分。最后综合判断，把结论写进决策记录。
./reviews/deep_9.md:690:绝对不要用宽泛匹配的 `pkill -f codex`，那会误杀 Mac 上的 Codex.app。要杀进程就精确匹配。
./reviews/deep_9.md:704:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_9.md:825:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
./reviews/deep_9.md:846:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reviews/deep_9.md:862:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reviews/deep_9.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reviews/deep_9.md:921:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reviews/deep_9.md:965:  pdftotext -f 1 -l 8 \""'$f" - 2>/dev/null | rg -n -i '"'"'^('"\\s*)((abstract|introduction|related work|method|methods|methodology|approach|proposed|experiments?|results|conclusion|i+\\.|[0-9]+\\.?\\s+[A-Z]|[IVX]+\\.?\\s+[A-Z]))' | head -80
./reviews/deep_9.md:994:31:proposed to inject the description-based textual embedding into
./reviews/deep_9.md:999:130:method uses learnable-based textual embeddings to boost the
./reviews/deep_9.md:1041:37:methods [4], [5], [6], TIReID utilizes diverse text descriptions
./reviews/deep_9.md:1043:120:method to automatically obtain accurate text descriptions.
./reviews/deep_9.md:1048:346:results and obtain dependable image-text pairs with label
./reviews/deep_9.md:1051:392:2 represents that the cosine similarity between text features and class
./reviews/deep_9.md:1052:394:3 indicates abnormal text features are detected
./reviews/deep_9.md:1056:557:vi and t i , t vhn and vthn are the hardest negative text/image features of vi and t i . ϕvij is the image-to-text
./reviews/deep_9.md:1102:我开始逐篇抽文本。第一步先看标题、引言里的问题设定，再读方法区间到实验前，避免只从摘要倒推。
./reviews/deep_9.md:1108:/bin/zsh -lc "pdftotext -f 1 -l 8 'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '60,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_9.md:1385:/bin/zsh -lc "pdftotext -f 1 -l 8 'DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf' - | nl -ba | sed -n '40,540p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_9.md:1391:    44	the Visual-Language Models(VLM), the textual embedding
./reviews/deep_9.md:1397:    50	the first method using identity-level textual space to align the
./reviews/deep_9.md:1401:    54	the textual-level identity-aware embeddings inferred from the
./reviews/deep_9.md:1402:    55	identity-aware prompts, denoted as the learnable-based textual
./reviews/deep_9.md:1403:    56	embedding. However, learnable-based textual embedding does
./reviews/deep_9.md:1404:    57	not capture the text description containing human priors,
./reviews/deep_9.md:1406:    59	of each person’s image. Recently, multiple types of textual
./reviews/deep_9.md:1408:    61	textual space [3], [8], [9], e.g., MP-ReID [3] firstly applies
./reviews/deep_9.md:1410:    63	fused with the learnable textual tokens and visual tokens by a
./reviews/deep_9.md:1413:    66	the learnable-based textual embedding and description-based
./reviews/deep_9.md:1414:    67	textual embedding, respectively. However, the synchronization strategy treats the learnable-based and description-based
./reviews/deep_9.md:1415:    68	textual embedding equally, leading to the generated visual
./reviews/deep_9.md:1417:    70	textual embedding, especially for the description-based textual
./reviews/deep_9.md:1420:    73	considering the benefits of learnable-based and descriptionbased textual embedding.
./reviews/deep_9.md:1421:    74	To address the above shortcomings, we introduce an asymmetrical strategy that fuses learnable-based and descriptionbased textual embedding with visual embedding. Although
./reviews/deep_9.md:1422:    75	description-based textual embedding can provide essential
./reviews/deep_9.md:1423:    76	clues containing the human prior, low-quality text description
./reviews/deep_9.md:1424:    77	would lead to the description-based textual embedding being
./reviews/deep_9.md:1425:    78	less discriminative. Note that low-quality images and meaningless descriptions like noise can easily generate low-quality text
./reviews/deep_9.md:1427:    80	the description-based textual embeddings as auxiliary clues
./reviews/deep_9.md:1428:    81	to boost visual and textual representation learning rather than
./reviews/deep_9.md:1429:    82	decisive information. Consequently, we inject the descriptionbased textual embedding into the learnable-based and visual
./reviews/deep_9.md:1434:    87	in textual description, the identity-aware description-based
./reviews/deep_9.md:1435:    88	textual embedding is generated by averaging the descriptionbased textual embeddings belonging to the same identity,
./reviews/deep_9.md:1436:    89	which is used to boost the discriminative to infer the learnablebased textual space used for aligning the visual representation
./reviews/deep_9.md:1439:    92	Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person representation learning, as shown in Figure 1. We utilize MLLM
./reviews/deep_9.md:1442:    95	Text Encoder to obtain description-based textual embeddings.
./reviews/deep_9.md:1443:    96	Meanwhile, similar to CLIP-ReID, we generate visual embedding and learnable-based textual embedding. After that,
./reviews/deep_9.md:1446:    99	textual embeddings with the description-based textual embeddings for injecting the human-level knowledge inferred from
./reviews/deep_9.md:1447:   100	the textual description. We then implement a cross-granularity
./reviews/deep_9.md:1449:   102	supervised using image-level description-based textual embeddings and visual embeddings, and image-level visual embeddings are supervised using identity-level description-based
./reviews/deep_9.md:1450:   103	textual embeddings. This method asymmetrically leverages the
./reviews/deep_9.md:1451:   104	description-based textual space and the learnable-based textual
./reviews/deep_9.md:1456:   109	consider the benefits of different types of textual embeddings, such as learnable-based and description-based
./reviews/deep_9.md:1457:   110	textual embeddings.
./reviews/deep_9.md:1458:   111	2) We propose a novel Dual Asymmetric Textual Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person
./reviews/deep_9.md:1473:   126	image-text pairs and has shown promising ability. CoOp [15]
./reviews/deep_9.md:1477:   130	method uses learnable-based textual embeddings to boost the
./reviews/deep_9.md:1488:   141	prompt learning strategy [15] to generate a set of discriminative text embeddings and then use it to enhance the visual space by minimizing InfoNCE loss [19] between visual
./reviews/deep_9.md:1489:   142	embeddings and textual embeddings. Instruct-ReID [9] uses
./reviews/deep_9.md:1495:   148	employed synchronous fusion of multiple modalities, overlooking the noise within the description-based textual embeddings. In our approach, we achieve a more effective model by
./reviews/deep_9.md:1496:   149	treating the description-based textual embeddings as auxiliary
./reviews/deep_9.md:1497:   150	clues to boost visual and textual representation learning.
./reviews/deep_9.md:1501:   154	Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person representation learning. As shown in Figure 1, the proposed
./reviews/deep_9.md:1503:   156	space, learnable-based textual space, and description-based
./reviews/deep_9.md:1504:   157	textual space. The description-based textual space first applies
./reviews/deep_9.md:1505:   158	MLLM to extract the textual description for each image and
./reviews/deep_9.md:1509:   162	At2t injects the description-based textual embedding into the
./reviews/deep_9.md:1510:   163	learnable-based textual embedding generated by feeding the
./reviews/deep_9.md:1516:   169	visual and textual embeddings inferred from the identity-aware
./reviews/deep_9.md:1517:   170	prompts, denoted as the learnable-based textual embedding.
./reviews/deep_9.md:1518:   171	However, learnable-based textual embedding does not capture
./reviews/deep_9.md:1519:   172	the text description containing human priors, leading to a
./reviews/deep_9.md:1654:   307	Fig. 1. The framework of DATE. DATE employs a description-based textual space as complementary to the visual and learnable-based textual space. We
./reviews/deep_9.md:1655:   308	use two cross-attention adapters to fuse description-based textual descriptions with visual and learnable-based textual embedding, respectively. We conduct ��鞋，特征相似度分布在正交化后更清晰，这些可视化都在支撑“语义分离是对的”。
./reassess/r_2.md:5958:reassess/r_3.md:8800:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:5965:reassess/r_3.md:8868:本地 GOPL kill-switch 设计已经明确把“2D 可见性解释同样强”列为致命红线；用户给出的新结果正好击中这条红线。遮挡 source-separation 只有一条很窄的生路：它必须把隐藏变量从“目标缺失多少”换成“遮挡者身份向量具体污染了谁”，并且压掉这部分后直接提高 ReID，而不是又回到 occlusion-count。
./reassess/r_2.md:5966:reassess/r_3.md:8885:一个做行人 ReID 的团队要发 CCF-B **方法稿**, 刚连续用零训练 kill-switch 干净杀掉两个方向:
./reassess/r_2.md:5969:reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_2.md:5973:reassess/r_3.md:9165:pivot/p_2.md:3325:./validate/v_3.md:15168:../litreview2/validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:5974:reassess/r_3.md:9166:pivot/p_2.md:3647:./validate/v_3.md:15494:../litreview2/validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:5975:reassess/r_3.md:9167:pivot/p_2.md:3955:./validate/v_3.md:16098:../litreview2/validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:5976:reassess/r_3.md:9197:pivot/p_2.md:6913:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:5977:reassess/r_3.md:9200:pivot/p_2.md:6924:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:5986:reassess/r_3.md:9219:pivot/p_2.md:7972:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:5989:reassess/r_3.md:9228:pivot/p_2.md:8154:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:5992:reassess/r_3.md:9249:validate/v_3.md:1887:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:5993:reassess/r_3.md:9253:validate/v_3.md:1906:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:5994:reassess/r_3.md:9256:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:5996:reassess/r_3.md:9261:validate/v_3.md:2945:../litreview2/analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6001:reassess/r_3.md:9307:validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:6002:reassess/r_3.md:9308:validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:6003:reassess/r_3.md:9312:validate/v_3.md:14184:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:6004:reassess/r_3.md:9313:validate/v_3.md:15168:../litreview2/validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:6005:reassess/r_3.md:9314:validate/v_3.md:15494:../litreview2/validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:6006:reassess/r_3.md:9315:validate/v_3.md:16098:../litreview2/validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:6010:reassess/r_3.md:9388:validate/v_2.md:5109:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6011:reassess/r_3.md:9389:validate/v_2.md:5121:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6012:reassess/r_3.md:9391:validate/v_2.md:5173:./analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6013:reassess/r_3.md:9395:validate/v_2.md:5932:./analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6014:reassess/r_3.md:9480:validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_2.md:6024:reassess/r_3.md:9843:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6028:reassess/r_3.md:10023:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6029:reassess/r_3.md:10028:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6033:reassess/r_3.md:10044:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6083:reassess/r_3.md:11155:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6086:reassess/r_3.md:11168:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6097:reassess/r_3.md:11273:pivot/p_2.md:6913:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6098:reassess/r_3.md:11275:pivot/p_2.md:6924:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6108:reassess/r_3.md:11307:pivot/p_2.md:7972:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6109:reassess/r_3.md:11309:pivot/p_2.md:8154:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6117:reassess/r_3.md:11335:validate/v_3.md:1887:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6118:reassess/r_3.md:11337:validate/v_3.md:1906:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6119:reassess/r_3.md:11341:validate/v_3.md:2945:../litreview2/analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6137:reassess/r_3.md:11418:reviews/deep_4.md:790:encapsulates the essential identity of the pedestrian. The attribute token A∗ and the context token O∗ serve as detail tokens, capturing identity-relevant attributes and occluding objects respectively. FLaN-Net learns detailed information for
./reassess/r_2.md:6148:reassess/r_3.md:11498:validate/v_2.md:5109:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6149:reassess/r_3.md:11499:validate/v_2.md:5121:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6150:reassess/r_3.md:11500:validate/v_2.md:5173:./analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6152:reassess/r_3.md:11502:validate/v_2.md:5932:./analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6160:reassess/r_3.md:11630:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6161:reassess/r_3.md:11632:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_2.md:6187:一个做行人 ReID 的团队要发 CCF-B **方法稿**, 刚连续用零训练 kill-switch 干净杀掉两个方向:
./reassess/r_2.md:6196:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_2.md:6239:-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
./reassess/r_2.md:6396:web search: "hubness" "person re-identification"
./reassess/r_2.md:6432:web search: hubness image retrieval deep features CSLS mutual proximity "image retrieval"
./reassess/r_2.md:6440:结论：我只推荐 **1 个主线 + 1 个备胎**。不要再救 SMPL / 遮挡 / 航拍-地面；也不要把“相机偏差”当主线，DART³ 已经把 camera bias + distance-aware TTA 写得很近，CA-Jaccard 也占了 camera-aware neighbor/re-ranking 这块。([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2311.10605))
./reassess/r_2.md:6458:廉价 kill-switch：
./reassess/r_2.md:6467:- k-reciprocal re-ranking 已经用 reciprocal neighbor 做 test-time ranking，但它不是把 **gallery in-degree hubness** 定义成强 ReID 的失败变量，也不是训练端 anti-hub。([arxiv.org](https://arxiv.org/abs/1701.08398))
./reassess/r_2.md:6469:- hubness 在通用高维检索里是成熟概念，甚至已有工作把 hubness 和 CSLS 当核心机制分析；好处是理论可借，风险是必须证明 ReID 里没人把它做成主问题。([arxiv.org](https://arxiv.org/abs/2605.26575))
./reassess/r_2.md:6486:零训练 kill-switch：
./reassess/r_2.md:6496:- 风险：这更像效率/协议稿，不是纯涨点稿。只有 kill-switch 显示 Pareto 很漂亮，才值得推进。
./reassess/r_2.md:6498:优先级：先测 **Gallery Hubness**。它最像你们现在能做的 B 类方法稿：强 backbone 冻结就能证伪，完全绕开 SMPL/遮挡/航拍，机制也能自然从隐藏变量长出来。方向 2 只在 hubness 不成立时作为新协议备胎。
./reassess/r_2.md:6501:结论：我只推荐 **1 个主线 + 1 个备胎**。不要再救 SMPL / 遮挡 / 航拍-地面；也不要把“相机偏差”当主线，DART³ 已经把 camera bias + distance-aware TTA 写得很近，CA-Jaccard 也占了 camera-aware neighbor/re-ranking 这块。([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2311.10605))
./reassess/r_2.md:6519:廉价 kill-switch：
./reassess/r_2.md:6528:- k-reciprocal re-ranking 已经用 reciprocal neighbor 做 test-time ranking，但它不是把 **gallery in-degree hubness** 定义成强 ReID 的失败变量，也不是训练端 anti-hub。([arxiv.org](https://arxiv.org/abs/1701.08398))
./reassess/r_2.md:6530:- hubness 在通用高维检索里是成熟概念，甚至已有工作把 hubness 和 CSLS 当核心机制分析；好处是理论可借，风险是必须证明 ReID 里没人把它做成主问题。([arxiv.org](https://arxiv.org/abs/2605.26575))
./reassess/r_2.md:6547:零训练 kill-switch：
./reassess/r_2.md:6557:- 风险：这更像效率/协议稿，不是纯涨点稿。只有 kill-switch 显示 Pareto 很漂亮，才值得推进。
./reassess/r_2.md:6559:优先级：先测 **Gallery Hubness**。它最像你们现在能做的 B 类方法稿：强 backbone 冻结就能证伪，完全绕开 SMPL/遮挡/航拍，机制也能自然从隐藏变量长出来。方向 2 只在 hubness 不成立时作为新协议备胎。
./reassess/r_3.md:14:一个做行人 ReID 的团队要发 CCF-B **方法稿**, 刚连续用零训练 kill-switch 干净杀掉两个方向:
./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_3.md:66:-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
./reassess/r_3.md:605:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp340c_randomtext/design.md
./reassess/r_3.md:767:我定位到主实验文档在 `/Users/abdslm/Desktop/SOLIDER-REID/experiments`，当前目录是一次 litreview/kill-switch 工作区。下一步我会读主线文档里与 exp109、exp255、近期 exp347/348 相关的判断，避免把一个已撞过的遮挡方向换名再提。
./reassess/r_3.md:777:> 注：本节默认记录各实验的**首次单 seed / 单 checkpoint**结果，用于保留搜索轨迹；是否能作为论文主结论，以文末 **4090 多种子验证** 为准。
./reassess/r_3.md:814:| 024-g | PDS+StopGrad noPSG global-only | 59.2% | 68.7% | 82.0% | 86.1% | mAP+2.6%, R1+2.2% | 单 seed 高点；后续 multi-seed 不支持“PSG 贡献很小”这一强结论 |
./reassess/r_3.md:839:| 032-g | PSG + Keypoint Pooling Only (global-only) | 59.8% | 70.0% | 81.7% | 85.4% | mAP+3.2%, R1+3.5% | 单 seed；支持“branch 不解释 global 提升”，但精确结论应以 030a multi-seed 为准 |
./reassess/r_3.md:844:| 035b | PSG+GCN score*visibility weight | 60.4% | 71.6% | 84.8% | 87.9% | mAP+3.8%, R1+5.1% | ❌ vs 035a: -0.7% mAP, -2.2% R1。当前只说明 `score*visibility` 未带来收益，不能上升为整条 visibility 路线结论 |
./reassess/r_3.md:932:| 143-eq | PSG+GCN+SASA (Skeleton-Aware Self-Attention, eq) | 61.1% | 73.7% | 85.1% | 88.5% | vs exp030a-eq: **0.0%/0.0%** | 🟡 完美中性。零参数骨架测地注意力偏置对最终结果无任何影响。与 KP-RPE(exp052) 结论一致：Swin window attention 的 RPE 已足够编码空间结构 |
./reassess/r_3.md:933:| 141-cvk | PSG+GCN+LPCS comp_ctx (cvk_residual) | 55.8% | 68.1% | 78.3% | 82.4% | — | ❌ LPCS comp_ctx 失败。competition-context 未改善排序。LPCS 训练 loss 严重干扰主学习，最终远低于 exp030a (-5.3% mAP) |
./reassess/r_3.md:1049:**执行结果**: exp002 结果 mAP 57.2% vs exp001 57.1%，几乎无差异。id_part 训练中期收敛更快但最终效果相同。**结论：归一化方式不是瓶颈，转向方案 B。**
./reassess/r_3.md:1076:**执行结果**: exp003 在 ep60 终止，mAP 50.2%（-6.4% vs baseline）。降低 global loss weight 严重伤害 backbone 特征质量。Part 分类器虽学得更快（id_part 2.08 vs exp001 ~3.3），但池化的 backbone 特征变差了。**结论：global 和 part 是共生关系，不能通过削弱 global 来强化 part。**
./reassess/r_3.md:1106:**执行结果**: exp004 PFM 是中性结果。mAP 与 exp001 part-only 相同（57.5%），R1 反而下降 0.8%。PFM 加速收敛但不改善最终表征。**结论：不要在同一处重复使用 pose 信息（PFM+part pooling 是冗余的）。**
./reassess/r_3.md:1190:**执行结果**: exp008 mAP 57.7%, R1 66.0%。**组合不叠加**，低于 PSG-only (58.3%/67.9%)。Part pooling 的 part_only 测试模式丢弃了 PSG 增强的 global feature，而 part features 本身无法匹配 PSG-global 的质量。**结论：backbone-level injection 是更有效的 pose 利用方式，post-hoc pooling 在 PSG 基础上没有增量价值。**
./reassess/r_3.md:1219:**执行结果**: exp009 mAP 58.3%, R1 67.2%, R5 81.2%, R10 85.2%。Multi-stage PSG (Stage 2+3) 与 single-stage (Stage 3 only) mAP 持平，R1 略低（-0.7%），R5/R10 略优（+0.4%/+0.3%），但增加了 156K 额外参数。**结论：Stage 2 PSG 无显著收益，pose spatial gating 在 Stage 3 已足够。后续聚焦于改进 PSG 机制本身，而非扩展注入范围。**
./reassess/r_3.md:1271:### 当前重审结论
./reassess/r_3.md:1407:   - CLIP frozen text prototypes + cross-attention + pose heatmap bias
./reassess/r_3.md:1587:- **关键结论**: 不要用 visibility 向量，用原始 pose 热图
./reassess/r_3.md:1599:- **结论**: 让 backbone 在特征提取过程中知道人体结构，比事后选择更有效
./reassess/r_3.md:1749:### PDS 实验结论
./reassess/r_3.md:1785:主文档确认了一个关键背景：遮挡方向已有大量“补全、可见性、结构关系、masking、metric alignment”负结果，且 exp109 一类结论确实在说可实现的遮挡处理很容易退化成 identity-conditioned oracle 不可达。现在我会专门检索本地 GOPL、source-separation、occluder-gate 相关材料，再做外部 novelty 对照。
./reassess/r_3.md:1938:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_8.md:128:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:1943:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12133:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
./reassess/r_3.md:1944:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12180:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
./reassess/r_3.md:1952:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:17:ROLES[1]="角色=**残酷判官**。不留情面: 连续2个cheap-kill负 + SMPL反复证伪 + 三大资产全弱, 这个团队到底**还有没有**能发B类方法稿的真路? 还是诚实答案='手里的empirical资产出不了干净B类method, 该换问题域/换数据/换打法'? 联网核查ReID 2024-2026 method空间。**别和稀泥**: 要么明确指1条最可能成的(带理由+廉价kill-switch), 要么明确说'没有, 该转去X'(X具体)。"
./reassess/r_3.md:1953:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:18:ROLES[2]="角色=**全新方向探子**。彻底放下SMPL/遮挡/航拍-地面三个失败区(不准碰)。团队只剩: 强Swin/SOLIDER backbone(可冻结当强特征源) + occluded_duke/market/MSMT/CARGO数据 + 22招式 + 读过167篇。用联网+招式提**1-2个真正新的B类方向**: 新问题定义或新观察(不是新模块), 每个带触发观察/重定义/机制怎么长/**廉价(最好零训练)kill-switch**/撞车核查。优先'用强backbone冻结特征就能验隐藏变量'的方向(像我们之前那种零训练kill-switch)。务实中文。"
./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
./reassess/r_3.md:1960:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/strategic_pivot.sh:21:ROLES[2]="角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。"
./reassess/r_3.md:1983:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/cvpb_containment_killswitch_design.md:44:无 ID label 泄漏，eval_dist 升序+same pid&cam junk removal 正确。唯一 Low: C3 用 per-dim mean 非 median(spec 本就写"均值方差"，已改 label 为 view-mean)。
./reassess/r_3.md:1984:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:128:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:1989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:1991:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:648:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:1992:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:664:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2008:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:149:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2013:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2015:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:282:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2016:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:298:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2024:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:356:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2025:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:357:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2034:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:869:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2036:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:1061:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2042:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:3419:前三篇的方法段已经能看出各自的“问题重写方式”：UP-Person 把全量微调改写成“知识保留不足”，HDNet 把跨模态差异拆成“相机、模态、身份冗余的层级污染”，MPT 把文本提示从静态辅助改写成“能和图像编码器互相强化的中间模态”。我继续补齐剩下两篇的方法段，并把消融和可视化也读出来，用来回答“证据闭环”。
./reassess/r_3.md:2052:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:308:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2056:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:512:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2058:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:704:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2063:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2065:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:846:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2066:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:862:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2074:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2075:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:921:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2091:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_1.md:118:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2149:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2151:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:527:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2152:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:543:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2160:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:601:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2161:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:602:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2167:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:648:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2169:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:840:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2174:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:7493:1. 触发观察：作者看到图文局部对齐在极端干扰下会失效。雨雪雾、强光、遮挡和相似外观物体会造成 visual hallucination，模型以为某个局部位置有对应文本，实际是伪真实位置，导致局部错配。
./reassess/r_3.md:2176:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:7546:1. 触发观察：作者看到图文局部对齐在极端干扰下会失效。雨雪雾、强光、遮挡和相似外观物体会造成 visual hallucination，模型以为某个局部位置有对应文本，实际是伪真实位置，导致局部错配。
./reassess/r_3.md:2178:/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_killswitch.py:3:# PASS: high << low/mid (或 full < gated-oracle) → confound 成立。FAIL: 各带同涨同跌 → 判死。
./reassess/r_3.md:2195:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:1409:本地旧文档显示了一条很长的“姿态/遮挡 ReID”探索史，且许多后续实验已经在反复验证“加结构/补全/metric/loss 很难形成新机制”。我会再读你这轮 CARGO empirical 的 kill-switch 和综合材料，判断是否只是单次失败还是路线失败。
./reassess/r_3.md:2206:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:1864:写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
./reassess/r_3.md:2212:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:1994:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:2223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
./reassess/r_3.md:2224:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2026:reid-paper 项目 D4 诊断**已确认 donor-leakage 人际混淆在强主干仍持续(≈3×物体干扰)**——kill-switch 其实已经过了。衍生"donor-leakage 反事实一致性"方法 6-7/10、待探针。**库读证明: 你这条自己的活口, 正好是整个领域的空白。**
./reassess/r_3.md:2231:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2045:**领域(167篇)+ 你自己的 D4 诊断 + 获胜配方 三方收敛到同一个角度: 同框入侵身份泄漏的 source-separation(非补全)。** 这是 reid-paper 项目最强活口, 也是全领域空白, 且自带打 wall-2 的对照 + 一晚的 kill-switch。下一步: 核对"主要人物归属"当年判死理由 → 不撞则跑探针 → 建 donor-leakage 反事实一致性方法。
./reassess/r_3.md:2244:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2185:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:2250:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2219:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:2256:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2258:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:195:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2259:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:211:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2267:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:269:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2268:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:270:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2274:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:382:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2278:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:588:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2280:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:780:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2314:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/p_2.txt:1:结论先说：我会把主线从“怎么对齐航拍-地面”转成“训练标签里的同一身份关系并不等价”。SMPL 不再当身份特征，也不再做 canonical/visibility 对齐，而是当 **可测的几何条件变量**，用来重排训练关系或做因果干预。
./reassess/r_3.md:2336:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/p_2.txt:62:先做方向 1。它最贴合当前失败链：不救 avg/MaxSim，不碰 visibility 红海，不再把 SMPL 当身份特征，而且 kill-switch 足够干净。方向 2 作为备选，只有在 surface ACE 非常集中时再推进，否则容易被审稿人归类为“又一个人体区域遮挡增强”。
./reassess/r_3.md:2352:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/p_3.txt:29:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:2367:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp330_compositional_dro/design.md:29:- **副判据（脚本外，单独 test.py 验，不进自动 kill-switch）**：standard Occ-Duke 真实遮挡 mAP，确认不伤主任务。kill-switch 脚本只跑主判据（Market 组合 GAP）。
./reassess/r_3.md:2368:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp330_compositional_dro/design.md:33:- **前置 GAP 检查（先跑 ERM）**：ERM 自己 held-out mean mAP 是否显著 < seen mean（存在组合 GAP）。若 ERM 无 GAP（held≈seen）→ 无 occluder-class 捷径可利用 → 整 bet NO-GO（合理早 kill，省 DRO）。有 GAP 才看 DRO 能否合上。
./reassess/r_3.md:2375:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:1:结论先说：**GOPL 作为“用 SMPL 共同可见人体表面 overlap 来定义同 ID 正边可信度，并据此做训练关系排序/采样/延迟约束”目前存活**。  
./reassess/r_3.md:2377:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:6:| 类别 | verdict | 杀伤点 | GOPL 切开点 | 信心 |
./reassess/r_3.md:2379:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:9:| (b) noisy-correspondence / noisy-label ReID | 存活 | [FAT + label distillation](https://arxiv.org/abs/1912.07863)、[Sample Uncertainty for UDA ReID](https://arxiv.org/abs/2012.08733)、[LUPerson-NL](https://arxiv.org/abs/2203.16533) 都处理 noisy labels / pseudo-label reliability；text-image 里 [RDE noisy correspondence](https://arxiv.org/abs/2308.09911) 处理图文配对不可靠。 | 它们的“不可靠”主要是标签错、伪标签错、跨模态配对错。GOPL 假设 ID 标签没错，但 **同 ID pair 的监督强度过强**，这是不同问题定义。 | 8/10 |
./reassess/r_3.md:2391:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:134:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2396:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2398:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:452:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2399:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:468:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2407:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:526:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2408:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:527:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2417:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:745:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2419:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:937:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2431:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:662:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2433:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:685:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2434:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:701:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2442:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:759:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2443:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:760:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2468:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:126:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2469:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:403:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2471:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:595:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2479:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:839:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2481:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:862:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2482:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:878:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:936:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2491:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:937:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2499:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:976:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:2529:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2531:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:516:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2532:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:532:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2540:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:590:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2541:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:591:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2556:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:11265:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:2557:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:11342:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:2566:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:298:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2574:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2576:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:585:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2577:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:601:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2585:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:659:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2586:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:660:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2594:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:699:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:2595:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:757:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2597:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:949:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2604:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:432:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:2606:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:624:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:2611:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:737:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2613:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:760:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2614:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:776:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:834:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2623:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:835:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2631:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:874:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:2658:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2660:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:516:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2661:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:532:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2669:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:590:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:591:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2689:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:145:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:556:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:579:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2700:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:595:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2713:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:6749:3. 机制怎么长出来：第一阶段冻结 CLIP 图文编码器，用 CoOp 学属性 prompt 的可学习 token，把这些 token 当属性伪标签，同时训练 attribute encoder 从图像特征预测这些 token。第二阶段用整体文本特征和属性伪标签双重引导图像编码器，再用 noise-mask 跳过遮挡导致的不可靠属性匹配。
./reassess/r_3.md:2717:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:6804:3. 机制怎么长出来：第一阶段冻结 CLIP 图文编码器，用 CoOp 学属性 prompt 的可学习 token，把这些 token 当属性伪标签，同时训练 attribute encoder 从图像特征预测这些 token。第二阶段用整体文本特征和属性伪标签双重引导图像编码器，再用 noise-mask 跳过遮挡导致的不可靠属性匹配。
./reassess/r_3.md:2723:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:139:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:2728:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:261:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2730:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:284:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:2731:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:300:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:2739:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:358:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:2740:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:359:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:2748:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:398:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:2758:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:6424:3. 机制怎么长出来：既然要同时处理图像、文本、图文，就自然长出图像编码器、文本编码器、多模态编码器。既然模态会缺失，就用图像掩码、文本掩码、图文联合掩码来预训练。`MMM` 不是单纯随机遮挡，而是强调行人区域，服务于“视觉语义嵌入要关注人本体”的叙事。
./reassess/r_3.md:2765:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:6501:3. 机制怎么长出来：既然要同时处理图像、文本、图文，就自然长出图像编码器、文本编码器、多模态编码器。既然模态会缺失，就用图像掩码、文本掩码、图文联合掩码来预训练。`MMM` 不是单纯随机遮挡，而是强调行人区域，服务于“视觉语义嵌入要关注人本体”的叙事。
./reassess/r_3.md:2775:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:26:角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。
./reassess/r_3.md:2776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:35:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./reassess/r_3.md:2778:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:44:./strategic_pivot.sh:21:ROLES[2]="角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。"
./reassess/r_3.md:2789:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:132:./pivot/p_3.md:1409:本地旧文档显示了一条很长的“姿态/遮挡 ReID”探索史，且许多后续实验已经在反复验证“加结构/补全/metric/loss 很难形成新机制”。我会再读你这轮 CARGO empirical 的 kill-switch 和综合材料，判断是否只是单次失败还是路线失败。
./reassess/r_3.md:2797:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:185:./pivot/p_3.md:1864:写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
./reassess/r_3.md:2799:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:193:./pivot/p_2.md:26:角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。
./reassess/r_3.md:2801:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:212:./analyses/deep_22.txt:9:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:2803:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:218:./reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2807:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:250:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2809:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:306:./reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2811:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:339:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2813:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:391:./reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2815:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:567:./reviews/deep_2.md:737:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2819:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:610:./reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2843:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:819:./B_CONTAINMENT_DESIGN.md:66:写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
./reassess/r_3.md:2848:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:835:./analyses/deep_4.txt:53:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:2849:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:836:./analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:2850:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:837:./analyses/deep_4.txt:57:3. 机制怎么长出来：机制基本是顺着这个重定义长出来的。主体 token S* 来自全局图像特征，表示“这是谁”；属性 token A* 由 learnable queries 从 patch 中提取可见身份属性；上下文 token O* 提取遮挡物特征。三者组成 “A photo of a [S*] person with [A*] partially occluded by [O*]” 这样的细粒度 prompt，再用文本作为 query 对视觉 patch 做 cross-attention，引导图像特征看可见人体区域、压掉遮挡物。动态融合用预测不确定性给 visual、text、cross-attention feature 分配权重，是对可靠性问题的补强。
./reassess/r_3.md:2860:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:859:./reviews/deep_13.md:662:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2862:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:900:./reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2865:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:956:./reviews/deep_17.md:261:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2882:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1029:./analyses/ALL_ANALYSES.txt:942:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:2889:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1061:./analyses/ALL_ANALYSES.txt:1524:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:2890:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1062:./analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:2891:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1063:./analyses/ALL_ANALYSES.txt:1528:3. 机制怎么长出来：机制基本是顺着这个重定义长出来的。主体 token S* 来自全局图像特征，表示“这是谁”；属性 token A* 由 learnable queries 从 patch 中提取可见身份属性；上下文 token O* 提取遮挡物特征。三者组成 “A photo of a [S*] person with [A*] partially occluded by [O*]” 这样的细粒度 prompt，再用文本作为 query 对视觉 patch 做 cross-attention，引导图像特征看可见人体区域、压掉遮挡物。动态融合用预测不确定性给 visual、text、cross-attention feature 分配权重，是对可靠性问题的补强。
./reassess/r_3.md:2895:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1078:./reviews/deep_12.md:431:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2901:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1127:./reviews/deep_5.md:806:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2903:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1176:./reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2905:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1213:./reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2953:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1308:./reviews/deep_20.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2955:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1333:./reviews/deep_11.md:467:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2957:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1396:./reviews/deep_24.md:839:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2959:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1445:./reviews/deep_4.md:183:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2960:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1493:./reviews/deep_4.md:11383:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:2961:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1494:./reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:2962:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1495:./reviews/deep_4.md:11387:3. 机制怎么长出来：机制基本是顺着这个重定义长出来的。主体 token S* 来自全局图像特征，表示“这是谁”；属性 token A* 由 learnable queries 从 patch 中提取可见身份属性；上下文 token O* 提取遮挡物特征。三者组成 “A photo of a [S*] person with [A*] partially occluded by [O*]” 这样的细粒度 prompt，再用文本作为 query 对视觉 patch 做 cross-attention，引导图像特征看可见人体区域、压掉遮挡物。动态融合用预测不确定性给 visual、text、cross-attention feature 分配权重，是对可靠性问题的补强。
./reassess/r_3.md:2965:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1500:./reviews/deep_4.md:11460:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:2966:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1501:./reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:2967:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1502:./reviews/deep_4.md:11464:3. 机制怎么长出来：机制基本是顺着这个重定义长出来的。主体 token S* 来自全局图像特征，表示“这是谁”；属性 token A* 由 learnable queries 从 patch 中提取可见身份属性；上下文 token O* 提取遮挡物特征。三者组成 “A photo of a [S*] person with [A*] partially occluded by [O*]” 这样的细粒度 prompt，再用文本作为 query 对视觉 patch 做 cross-attention，引导图像特征看可见人体区域、压掉遮挡物。动态融合用预测不确定性给 visual、text、cross-attention feature 分配权重，是对可靠性问题的补强。
./reassess/r_3.md:2971:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1506:./reviews/deep_10.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2986:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1727:./reviews/deep_14.md:673:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2994:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1801:./reviews/deep_3.md:556:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:2999:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1896:./reviews/deep_22.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3000:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2023:./reviews/deep_22.md:11265:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3001:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2029:./reviews/deep_22.md:11342:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3013:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2144:./validate/v_2.md:4044:./reviews/deep_22.md:11342:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3023:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2185:./validate/v_2.md:5132:./analyses/deep_22.txt:9:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3032:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2212:./validate/v_2.md:5892:./analyses/ALL_ANALYSES.txt:942:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3043:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2275:./validate/v_2.md:7023:validate/v_3.md:2882:../litreview2/analyses/ALL_ANALYSES.txt:942:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3044:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2277:./validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:3045:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2278:./validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:3047:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2282:./validate/v_2.md:7181:analyses/deep_22.txt:9:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3049:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2286:./validate/v_2.md:7270:analyses/ALL_ANALYSES.txt:942:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3050:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2316:./validate/v_2.md:8272:validate/v_2.md:4044:./reviews/deep_22.md:11342:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3053:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2321:./validate/v_2.md:8613:validate/v_2.md:5132:./analyses/deep_22.txt:9:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3056:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2326:./validate/v_2.md:8774:validate/v_2.md:5892:./analyses/ALL_ANALYSES.txt:942:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3061:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2379:./validate/v_2.md:11572:reviews/deep_22.md:11265:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3062:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2382:./validate/v_2.md:11577:reviews/deep_22.md:11342:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3077:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2939:./validate/v_3.md:14782:../litreview2/validate/v_3.md:6982:./validate/v_3.md:2364:../codex_review_ovli.txt:3535:experiments/MORNING_REPORT.md:63:3. **两轮真调研（联网 + 对抗验证，30+ agent / 300万 token）**：现有 occluded 上 8 候选 **0 过审**（每个有真实顶会先例 + 撞墙，有据可查）；TBPS + 亲缘任务 7 候选**活 1 个 = PartNC**（用 pose 可见性区分"文本-图对不上=遮挡 vs =标注错"，复用我们 part-MaxSim + 遮挡老本）。**诚实：PartNC 非稳赢，是探索性赌注，但有干净的 2-3 天廉价 kill-switch。**
./reassess/r_3.md:3115:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3346:./validate/v_3.md:15189:../litreview2/validate/v_3.md:7389:./validate/v_3.md:2775:../codex_review_ovli.txt:3980:experiments/overnight_innovation_log.md:389:**唯一幸存 PartNC(Part-level Noisy-Correspondence)**：用部位粒度 image-text 相似度估每个身体部位 clean/noisy 置信(RDE 的 CCD 从 pair 级下沉到部位级)；**真正空白=用 pose-visibility 先验区分"相似度低=遮挡(结构化)"vs"=标注错(随机)"**(RDE/DURA/GA-DMS 把 noise 当随机，表达不出)。复用我们 part-MaxSim 逻辑(einsum bkc,gpc->bgkp .max)+pose 可见性积累。
./reassess/r_3.md:3131:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3612:./validate/v_3.md:15459:../litreview2/validate/v_3.md:8015:./validate/v_2.md:12689:validate/v_3.md:2364:../codex_review_ovli.txt:3535:experiments/MORNING_REPORT.md:63:3. **两轮真调研（联网 + 对抗验证，30+ agent / 300万 token）**：现有 occluded 上 8 候选 **0 过审**（每个有真实顶会先例 + 撞墙，有据可查）；TBPS + 亲缘任务 7 候选**活 1 个 = PartNC**（用 pose 可见性区分"文本-图对不上=遮挡 vs =标注错"，复用我们 part-MaxSim + 遮挡老本）。**诚实：PartNC 非稳赢，是探索性赌注，但有干净的 2-3 天廉价 kill-switch。**
./reassess/r_3.md:3158:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3920:./validate/v_3.md:16063:../litreview2/validate/v_2.md:12689:validate/v_3.md:2364:../codex_review_ovli.txt:3535:experiments/MORNING_REPORT.md:63:3. **两轮真调研（联网 + 对抗验证，30+ agent / 300万 token）**：现有 occluded 上 8 候选 **0 过审**（每个有真实顶会先例 + 撞墙，有据可查）；TBPS + 亲缘任务 7 候选**活 1 个 = PartNC**（用 pose 可见性区分"文本-图对不上=遮挡 vs =标注错"，复用我们 part-MaxSim + 遮挡老本）。**诚实：PartNC 非稳赢，是探索性赌注，但有干净的 2-3 天廉价 kill-switch。**
./reassess/r_3.md:3171:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4077:./reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3195:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4224:本地材料确认了 containment 的三项 kill-switch 已经写入文档，且 `SYNTHESIS_METHODOLOGY.md` 里也把 SMPL canonical-view 记为高撞车风险。下一步我会把联网核查重点放在 2024-2026 的 aerial-ground/SMPL/ReID 邻近方法，而不是再搜 avg/MaxSim。
./reassess/r_3.md:3205:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4500:角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。
./reassess/r_3.md:3206:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4509:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./reassess/r_3.md:3208:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4518:./strategic_pivot.sh:21:ROLES[2]="角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。"
./reassess/r_3.md:3219:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4606:./pivot/p_3.md:1409:本地旧文档显示了一条很长的“姿态/遮挡 ReID”探索史，且许多后续实验已经在反复验证“加结构/补全/metric/loss 很难形成新机制”。我会再读你这轮 CARGO empirical 的 kill-switch 和综合材料，判断是否只是单次失败还是路线失败。
./reassess/r_3.md:3227:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4659:./pivot/p_3.md:1864:写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
./reassess/r_3.md:3229:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4667:./pivot/p_2.md:26:角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。
./reassess/r_3.md:3231:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4686:./analyses/deep_22.txt:9:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:3233:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4692:./reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3237:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4731:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3239:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4787:./reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3241:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4820:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3243:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4872:./reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3252:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5043:reviews/deep_8.md:12133:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
./reassess/r_3.md:3253:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5045:reviews/deep_8.md:12180:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
./reassess/r_3.md:3265:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5105:pivot/p_3.md:1409:本地旧文档显示了一条很长的“姿态/遮挡 ReID”探索史，且许多后续实验已经在反复验证“加结构/补全/metric/loss 很难形成新机制”。我会再读你这轮 CARGO empirical 的 kill-switch 和综合材料，判断是否只是单次失败还是路线失败。
./reassess/r_3.md:3275:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5127:pivot/p_3.md:1864:写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
./reassess/r_3.md:3281:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5136:pivot/p_3.md:1994:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:3296:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_1.md:23:角色=**残酷判官**。不留情面: 连续2个cheap-kill负 + SMPL反复证伪 + 三大资产全弱, 这个团队到底**还有没有**能发B类方法稿的真路? 还是诚实答案='手里的empirical资产出不了干净B类method, 该换问题域/换数据/换打法'? 联网核查ReID 2024-2026 method空间。**别和稀泥**: 要么明确指1条最可能成的(带理由+廉价kill-switch), 要么明确说'没有, 该转去X'(X具体)。
./reassess/r_3.md:3309:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_7.md:8446:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp330_compositional_dro/design.md:33:- **前置 GAP 检查（先跑 ERM）**：ERM 自己 held-out mean mAP 是否显著 < seen mean（存在组合 GAP）。若 ERM 无 GAP（held≈seen）→ 无 occluder-class 捷径可利用 → 整 bet NO-GO（合理早 kill，省 DRO）。有 GAP 才看 DRO 能否合上。
./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_3.md:3339:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:767:我定位到主实验文档在 `/Users/abdslm/Desktop/SOLIDER-REID/experiments`，当前目录是一次 litreview/kill-switch 工作区。下一步我会读主线文档里与 exp109、exp255、近期 exp347/348 相关的判断，避免把一个已撞过的遮挡方向换名再提。
./reassess/r_3.md:3351:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:1785:主文档确认了一个关键背景：遮挡方向已有大量“补全、可见性、结构关系、masking、metric alignment”负结果，且 exp109 一类结论确实在说可实现的遮挡处理很容易退化成 identity-conditioned oracle 不可达。现在我会专门检索本地 GOPL、source-separation、occluder-gate 相关材料，再做外部 novelty 对照。
./reassess/r_3.md:3366:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3368:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:518:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3369:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:534:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3385:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:3611:区别：最像 `FMFA`，但它解决文本和图像的显式局部对齐；我们解决图像 ReID 里“同身份正样本局部不可见或错位”的问题，用 SMPL 投影建立可验证的正样本 token 对齐矩阵。
./reassess/r_3.md:3394:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:3676:区别：最像 `FMFA`，但它解决文本和图像的显式局部对齐；我们解决图像 ReID 里“同身份正样本局部不可见或错位”的问题，用 SMPL 投影建立可验证的正样本 token 对齐矩阵。
./reassess/r_3.md:3398:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/new_angle_AIRL.md:84:## ★ kill-switch #3 结果(零训练 gate/fusion oracle, codex 红队设计, 2026-06-23)= 待填
./reassess/r_3.md:3404:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/new_angle_AIRL.md:169:2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一致(同一 degrade+forward 原语), 刻意保持对齐以保证消融诚实; 是否有实质影响由 kill-switch #4 训练结果裁决, 非 bug。若要彻底隔离需加 rec-only forward(改训练行为, 当前 lab-3090 正在跑此代码, 不动)。
./reassess/r_3.md:3407:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:23:角色=**全新方向探子**。彻底放下SMPL/遮挡/航拍-地面三个失败区(不准碰)。团队只剩: 强Swin/SOLIDER backbone(可冻结当强特征源) + occluded_duke/market/MSMT/CARGO数据 + 22招式 + 读过167篇。用联网+招式提**1-2个真正新的B类方向**: 新问题定义或新观察(不是新模块), 每个带触发观察/重定义/机制怎么长/**廉价(最好零训练)kill-switch**/撞车核查。优先'用强backbone冻结特征就能验隐藏变量'的方向(像我们之前那种零训练kill-switch)。务实中文。
./reassess/r_3.md:3411:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:143:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_3.md:3432:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2811:cheap kill-switch：先做纯评测级首验。提取全局特征和按 SMPL 区域聚合的局部特征，对每个 query-gallery 对取可见区域 soft-min 距离或 visibility-penalized 距离。若不训练都没有稳定改善，说明“最优几何距离”信号弱，不继续写复杂损失。
./reassess/r_3.md:3436:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2816:cheap kill-switch：先做三种极简注入：按姿态热图遮挡非关键区域、按 SMPL 可见性 drop 掉地面图像的部分身体区域、把航拍背景块注入地面图。只跑 Swin-Tiny 小规模快跑。如果损失正常但验证集无一致提升，说明正噪声假设不成立。
./reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_3.md:3447:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2871:cheap kill-switch：先做纯评测级首验。提取全局特征和按 SMPL 区域聚合的局部特征，对每个 query-gallery 对取可见区域 soft-min 距离或 visibility-penalized 距离。若不训练都没有稳定改善，说明“最优几何距离”信号弱，不继续写复杂损失。
./reassess/r_3.md:3451:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2876:cheap kill-switch：先做三种极简注入：按姿态热图遮挡非关键区域、按 SMPL 可见性 drop 掉地面图像的部分身体区域、把航拍背景块注入地面图。只跑 Swin-Tiny 小规模快跑。如果损失正常但验证集无一致提升，说明正噪声假设不成立。
./reassess/r_3.md:3460:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:176:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3462:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:199:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3463:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:215:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3471:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:273:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3472:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:274:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:3496:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3256:“多分支、多粒度、多尺度”本身已经很拥挤。能发 B 类的不是再拼一个分支，而是先指出一个被忽略的真实失败机制，再让机制服务这个问题。对我们来说，最值得借的是三类写法：显式对齐、可靠性建模、源域或视角不平衡。衣服文本和普通多分支换衣方向不建议正面进入，除非能被 SMPL 或航拍地面的几何问题重新定义。
./reassess/r_3.md:3510:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3331:“多分支、多粒度、多尺度”本身已经很拥挤。能发 B 类的不是再拼一个分支，而是先指出一个被忽略的真实失败机制，再让机制服务这个问题。对我们来说，最值得借的是三类写法：显式对齐、可靠性建模、源域或视角不平衡。衣服文本和普通多分支换衣方向不建议正面进入，除非能被 SMPL 或航拍地面的几何问题重新定义。
./reassess/r_3.md:3533:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324h/design.md:41:- **若仍冗余**：oracle ~+0.1、P_only ~0.2% → adapted-DINO 也被 Swin 包含，确认 analysis 结论。
./reassess/r_3.md:3580:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:322:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3582:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:345:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3583:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:361:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3591:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:419:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3592:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:420:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:3602:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:2174:前 5 篇里已经出现几个高频模式：遮挡论文还在围绕“局部对齐加遮挡权重”，红外论文围绕“跨模态生成或伪标签净化”，泛化论文把“困难样本”重新定义成度量关系被扰动的样本。后面 4 篇我继续按同样口径抽取，最后只保留对我们能发 B 类有用的反推结论。
./reassess/r_3.md:3609:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3465:cheap kill-switch：零训练先用现有 SOLIDER token 加 SMPL 可见表面 mask，在 AG-ReID.v2 或 CARGO 上只看大视角差、低可见重叠子集。如果共同可见区域相似度不能减少明显误匹配，这条先降级。
./reassess/r_3.md:3611:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3470:cheap kill-switch：冻结 SOLIDER，训练很小的两头线性探针，一个头吃 SMPL 几何或可见性统计，一个头吃视觉残差。若 base-only 和 detail-only 在 hard subset 上没有互补性，融合也吃不掉原 SOLIDER 的错误，这条不要硬做。
./reassess/r_3.md:3615:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3475:cheap kill-switch：按 SMPL 可见性做 token dropout 或 surface dropout，小规模训练 5 到 10 个 epoch。必须证明它比随机擦除、随机 token dropout 更能改善极端视角子集，否则只是普通增广。
./reassess/r_3.md:3617:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3482:我建议优先压第 1 和第 2。第 1 的问题定义最清楚，和团队的 pose gating、SMPL、aerial-ground 都能挂上；第 2 的论文叙事更像 Pattern Recognition/TMM，能把“不要过度对齐”讲成机制。第 3 可以作为训练机制配套，第 4 更像无监督扩展，除非 cheap kill-switch 很强，否则不要先押。
./reassess/r_3.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3521:cheap kill-switch：零训练先用现有 SOLIDER token 加 SMPL 可见表面 mask，在 AG-ReID.v2 或 CARGO 上只看大视角差、低可见重叠子集。如果共同可见区域相似度不能减少明显误匹配，这条先降级。
./reassess/r_3.md:3624:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3526:cheap kill-switch：冻结 SOLIDER，训练很小的两头线性探针，一个头吃 SMPL 几何或可见性统计，一个头吃视觉残差。若 base-only 和 detail-only 在 hard subset 上没有互补性，融合也吃不掉原 SOLIDER 的错误，这条不要硬做。
./reassess/r_3.md:3628:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3531:cheap kill-switch：按 SMPL 可见性做 token dropout 或 surface dropout，小规模训练 5 到 10 个 epoch。必须证明它比随机擦除、随机 token dropout 更能改善极端视角子集，否则只是普通增广。
./reassess/r_3.md:3630:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3538:我建议优先压第 1 和第 2。第 1 的问题定义最清楚，和团队的 pose gating、SMPL、aerial-ground 都能挂上；第 2 的论文叙事更像 Pattern Recognition/TMM，能把“不要过度对齐”讲成机制。第 3 可以作为训练机制配套，第 4 更像无监督扩展，除非 cheap kill-switch 很强，否则不要先押。
./reassess/r_3.md:3644:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:339:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3646:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:362:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3647:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:378:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3655:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:436:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3656:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:437:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:3714:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:322:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3716:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:345:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3717:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:361:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3725:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:419:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3726:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:420:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:3741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3165:cheap kill-switch：不用训练新模型，先在 CARGO 或 AG-ReID.v2 上用现有 SMPL/pose 估计，把 SOLIDER token 按身体区域重聚合，比较原始全局特征和 canonical part 特征的跨视角同 ID 相似度。如果同 ID 排名或正负间隔没有明显改善，这条先停。
./reassess/r_3.md:3745:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3174:区别：最像 DASA 和 DCR。DASA 用预训练卷积语义和 BN 适配，DCR 用属性文本做一致性桥。我们的切开点是 3D 几何坐标，它不是属性词，也不是旧分类器，而是跨域、跨视角、跨衣着都相对稳定的中间表示。机制可以是冻结 SOLIDER 主干，只训练 geometry-conditioned adapter，让旧域知识以 SMPL body surface token 或 joint-token 原型保存。  
./reassess/r_3.md:3746:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3175:cheap kill-switch：在现有顺序数据上做一个轻量探针，比较普通 SOLIDER 特征和 SMPL 分区后的特征在旧域遗忘样本上的互补性。如果 SMPL 分区特征不能挽回旧域近邻，或者和原特征高度冗余，这条不押。
./reassess/r_3.md:3749:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3180:cheap kill-switch：先做零训练分析，按估计视角高度或 SMPL 投影误差分桶，看跨域检索错误是否随这些物理量单调变化。没有单调关系，就说明物理视角约束不是主因，别硬做。
./reassess/r_3.md:3753:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3243:cheap kill-switch：不用训练新模型，先在 CARGO 或 AG-ReID.v2 上用现有 SMPL/pose 估计，把 SOLIDER token 按身体区域重聚合，比较原始全局特征和 canonical part 特征的跨视角同 ID 相似度。如果同 ID 排名或正负间隔没有明显改善，这条先停。
./reassess/r_3.md:3757:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3252:区别：最像 DASA 和 DCR。DASA 用预训练卷积语义和 BN 适配，DCR 用属性文本做一致性桥。我们的切开点是 3D 几何坐标，它不是属性词，也不是旧分类器，而是跨域、跨视角、跨衣着都相对稳定的中间表示。机制可以是冻结 SOLIDER 主干，只训练 geometry-conditioned adapter，让旧域知识以 SMPL body surface token 或 joint-token 原型保存。  
./reassess/r_3.md:3758:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3253:cheap kill-switch：在现有顺序数据上做一个轻量探针，比较普通 SOLIDER 特征和 SMPL 分区后的特征在旧域遗忘样本上的互补性。如果 SMPL 分区特征不能挽回旧域近邻，或者和原特征高度冗余，这条不押。
./reassess/r_3.md:3761:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:3258:cheap kill-switch：先做零训练分析，按估计视角高度或 SMPL 投影误差分桶，看跨域检索错误是否随这些物理量单调变化。没有单调关系，就说明物理视角约束不是主因，别硬做。
./reassess/r_3.md:3767:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:477:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:3772:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:587:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3774:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:610:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3775:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:626:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3783:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:684:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3784:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:685:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:3792:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2843:为什么能发：把图文 ReID 的局部错配定义成“局部可解性”问题，极端天气、光照、遮挡会让文本和图像局部根本不对应。机制是实体、属性、外观三阶段掩码重建。
./reassess/r_3.md:3798:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2916:为什么能发：把图文 ReID 的局部错配定义成“局部可解性”问题，极端天气、光照、遮挡会让文本和图像局部根本不对应。机制是实体、属性、外观三阶段掩码重建。
./reassess/r_3.md:3805:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:149:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:3813:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:419:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3815:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:442:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3816:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:458:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3824:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:516:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3825:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:517:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:3838:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3483:区别：最像 CVAF，但 CVAF 用 CLIP 文本 token 做视角一致语义过滤；我们用 SMPL 可见表面和投影关系定义哪些身体区域在航拍和地面视角中可比，避免文本语义和人工属性不可靠。
./reassess/r_3.md:3839:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3484:cheap kill-switch：在 CARGO 或 AG-ReID.v2 上用冻结 SOLIDER 特征，加 2D pose 或 SMPL 投影做可见表面池化。若零训练或小头训练不能明显改善跨视角正样本相似度和 mAP，直接停。
./reassess/r_3.md:3842:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3494:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
./reassess/r_3.md:3846:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3556:区别：最像 CVAF，但 CVAF 用 CLIP 文本 token 做视角一致语义过滤；我们用 SMPL 可见表面和投影关系定义哪些身体区域在航拍和地面视角中可比，避免文本语义和人工属性不可靠。
./reassess/r_3.md:3847:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3557:cheap kill-switch：在 CARGO 或 AG-ReID.v2 上用冻结 SOLIDER 特征，加 2D pose 或 SMPL 投影做可见表面池化。若零训练或小头训练不能明显改善跨视角正样本相似度和 mAP，直接停。
./reassess/r_3.md:3850:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3567:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
./reassess/r_3.md:3853:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_gate_oracle.py:3:AIRL fusion -- ZERO-TRAINING oracle kill-switch (codex red-team design).
./reassess/r_3.md:3855:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_gate_oracle.py:32:    the test-optimal threshold purely as an oracle ceiling for context.
./reassess/r_3.md:3860:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_gate_oracle.py:251:    print('AIRL fusion -- ZERO-TRAINING oracle kill-switch')
./reassess/r_3.md:3861:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_gate_oracle.py:401:    # area-gate oracle ceiling: best single area threshold on test (context only)
./reassess/r_3.md:3874:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:369:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3888:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:3369:cheap kill-switch：不训练模型，先用现有 SMPL 或 2D pose 估计在 CARGO 或 AG-ReID.v2 上算可见性兼容分数。若同身份跨视角对的可见性兼容分数不能明显高于困难负样本，或者把该分数加到 SOLIDER 距离后 mAP 没有超过 0.4 的净增益，就先杀掉。
./reassess/r_3.md:3893:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:3379:cheap kill-switch：先做零训练或浅层探针。单独用 SMPL 关节比例、投影姿态、步态统计做检索，如果它对同身份没有独立信号，或者和 SOLIDER late fusion 不能带来超过噪声线的增益，就不要开大训练。
./reassess/r_3.md:3900:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:3443:cheap kill-switch：不训练模型，先用现有 SMPL 或 2D pose 估计在 CARGO 或 AG-ReID.v2 上算可见性兼容分数。若同身份跨视角对的可见性兼容分数不能明显高于困难负样本，或者把该分数加到 SOLIDER 距离后 mAP 没有超过 0.4 的净增益，就先杀掉。
./reassess/r_3.md:3905:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:3453:cheap kill-switch：先做零训练或浅层探针。单独用 SMPL 关节比例、投影姿态、步态统计做检索，如果它对同身份没有独立信号，或者和 SOLIDER late fusion 不能带来超过噪声线的增益，就不要开大训练。
./reassess/r_3.md:3910:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:312:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:3912:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:504:如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。
./reassess/r_3.md:3917:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:619:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3919:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:642:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3920:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:658:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3928:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:716:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3929:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:717:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:3949:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3648:和最像工作的区别：MIP 的 prompt 是模态和实例向量，DTST 是 learned token 选择，π-VL 是 parsing 文本监督。我们的切开点是显式建模“这张图哪些人体证据可靠”，由 pose 置信度、遮挡、视角、mesh 可见性生成 reliability prompt 或 gating，指导 Swin 只强化可信身体证据。  
./reassess/r_3.md:3961:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3702:和最像工作的区别：MIP 的 prompt 是模态和实例向量，DTST 是 learned token 选择，π-VL 是 parsing 文本监督。我们的切开点是显式建模“这张图哪些人体证据可靠”，由 pose 置信度、遮挡、视角、mesh 可见性生成 reliability prompt 或 gating，指导 Swin 只强化可信身体证据。  
./reassess/r_3.md:3986:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:318:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:3988:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:341:**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。
./reassess/r_3.md:3989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:357:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
./reassess/r_3.md:3997:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:415:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:3998:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:416:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
./reassess/r_3.md:4004:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:518:- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
./reassess/r_3.md:4010:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2670:cheap kill-switch：先不训练。用现有 SMPL 或 2D pose 估计，把图像切成可见身体区域，测试 CARGO 或 AG-ReID.v2 上“几何可比区域相似度”是否比全局 SOLIDER 相似度更能区分正负样本。若同身份跨视角失败样本主要集中在低可见表面重叠区域，并且简单几何重加权能带来超过噪声线的提升，再立项。
./reassess/r_3.md:4012:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2680:cheap kill-switch：冻结 SOLIDER，用 pose heatmap 置信度或 SMPL 可见性做零训练 token reweighting。看航拍 query 对地面 gallery 的错误检索是否被低置信区域主导。若只用高置信身体区域能稳定改善，说明“区域可靠性”是真问题。
./reassess/r_3.md:4017:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2700:cheap kill-switch：先做训练很轻的辅助头。冻结或半冻结 SOLIDER，在现有数据上预测 SMPL joint visibility、身体表面区域或姿态几何 token，看辅助监督是否能改善跨视角检索。若只学几何 token 不改善，说明几何语义没有进入身份判别，及时停。
./reassess/r_3.md:4021:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2746:cheap kill-switch：先不训练。用现有 SMPL 或 2D pose 估计，把图像切成可见身体区域，测试 CARGO 或 AG-ReID.v2 上“几何可比区域相似度”是否比全局 SOLIDER 相似度更能区分正负样本。若同身份跨视角失败样本主要集中在低可见表面重叠区域，并且简单几何重加权能带来超过噪声线的提升，再立项。
./reassess/r_3.md:4023:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2756:cheap kill-switch：冻结 SOLIDER，用 pose heatmap 置信度或 SMPL 可见性做零训练 token reweighting。看航拍 query 对地面 gallery 的错误检索是否被低置信区域主导。若只用高置信身体区域能稳定改善，说明“区域可靠性”是真问题。
./reassess/r_3.md:4028:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2776:cheap kill-switch：先做训练很轻的辅助头。冻结或半冻结 SOLIDER，在现有数据上预测 SMPL joint visibility、身体表面区域或姿态几何 token，看辅助监督是否能改善跨视角检索。若只学几何 token 不改善，说明几何语义没有进入身份判别，及时停。
./reassess/r_3.md:4035:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/monitor.md:255:- ⚠️ **核心 kill-switch 隐忧**: AIRL 双分支论点 = f_rec(resolution-degradation consistency 训)学到**互补**的 resolvability-aware 表示, 融合后 ≥+1.0。ep10 两头**坍缩成同一表示**, 无互补证据可融。oracle 预言 +1.46, 实测 +0.06。
./reassess/r_3.md:4038:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp349_small_full_clip/claude_review.md:58:PLBOA 是数据增强（下半身遮挡），作用在输入图像/student view 上，OA-SD 用 clean（pre-PLBOA）图像与 pose 喂 teacher（line 819-824 `teacher_pose = pose_dict.get('teacher_pose', pose_dict)`，line 557 `img_teacher = img[3]` 为 clean view）。CLIP prompt 对齐的是 student 的 global_feat（带 PLBOA 遮挡）到 per-ID 文本原型 —— 这反而是「让被遮挡图也对齐纯 ID 语义」，与 PLBOA 目标方向一致、不冲突。✓ 纯增强，与 CLIP loss 计算路径无耦合。
./reassess/r_3.md:4045:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:7:- **机制**：CLIP-based（冻结 ViT-B/16 图像+文本编码器）。每张遮挡图做 textual inversion 构造细粒度 prompt `"A photo of a [S*] person with [A*] partially occluded by [O*]"`：S*=主体身份 token、A*=可见属性 token（衣着/配饰，learnable queries+cross-attn）、**O*=遮挡物 token（显式在语言空间建模遮挡物本身）**。cross-attn 对齐 prompt 与可见区域 + 熵不确定性加权融合 {视觉,文本,cross-attn}。
./reassess/r_3.md:4048:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:13:  1. **DMPA**（Dual-Masked Prompt Augmentation）：mask 可学**文本 prompt** token（α=0.5 置零）造两个 masked 版本，inter-text 对比。把遮挡视为**文本侧原型的 partial-information condition** → 遮挡图对齐到稳定文本锚。
./reassess/r_3.md:4052:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:22:4. **没试过的真新点**：(a) 遮挡物 token 语言建模（FLaN-Net O*）；(b) 文本-prompt masking 当遮挡代理（RMPSNet DMPA）；(c) 特征级区域优先 token 擦除（RMPSNet RPE 特征侧）。多为 CLIP-specific，但 (c) 可移植到任何栈。
./reassess/r_3.md:4056:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:48:- **Bet 2 = Gallery-as-target completion**：用检索到的 gallery 特征当补全目标(检索-补全反馈)。**廉价 kill-switch=oracle 天花板**: 把遮挡 query 特征换成同 ID 最高可见 gallery 特征重评，≥+4 mAP 过。**直接测 exp109"headroom is a wall"——大概率又是墙，0 训练可证否。**
./reassess/r_3.md:4062:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:78:| SCI | CLIP 文本正交投影减 occluder 方向 | = **FM-import CLIP** + 特征对齐 |
./reassess/r_3.md:4069:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_8.md:88:这个 bundle 里没有 `experiments/` 文档目录，只有 `code/`、`logs/` 和已有 review 文件。我先不读其他审查员结论，避免把独立判断污染掉。
./reassess/r_3.md:4091:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp323/design.md:44:**强模型当 baseline 不对**：GPT-5.5（codex）太强，可能自己已补上遮挡缺口 → B≈A → **假阴性**（天花板效应，不能 kill idea）；且方法真正部署/微调/蒸馏的是**小开源 MLLM**，GPT-5.5 在方法里只能当 oracle/teacher。
./reassess/r_3.md:4092:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp323/design.md:47:- **被测主体 = 小开源模型**（Qwen2.5-VL-**3B**/7B）——遮挡缺口真实存在、且是可部署 regime。**决定性 kill-switch 在这里**。
./reassess/r_3.md:4096:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:4:代码层面我没看到 AIRL eval/fuse 的关键实现错误，也没看到 FUSE 使用 query/gallery label 的泄漏。  
./reassess/r_3.md:4097:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:10:  训练中用 query/gallery eval 结果保存 `model_best`，若论文报告 best 而不是 final，会构成 test-set model selection 泄漏。当前 AG-ReID.v2 4090 主日志 best=final epoch60，所以这次主数未被实际污染。
./reassess/r_3.md:4098:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:42:- 主要风险不是代码泄漏，而是实验口径：不要报告 test-best；AG-ReID.v2 当前只能 claim “融合互补”，不能 claim “净超 baseline”。
./reassess/r_3.md:4099:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:149:degradation 实现是：只对 ground 样本做 normalized tensor 空间的 bilinear downsample 再 upsample，`s~U(0.25,1)`，无 blur；日志 `deg_scale_mean≈0.624/0.625` 正常。consistency 是 clean logits detach 作为 teacher，degraded rec logits 做 KL，`tau=4`，乘 `tau^2`，形式上是标准 KD/consistency loss。iso 下 `rec_only=True` 会在 stage-3 fork 处 detach，f_full BNNeck 不跑 degraded 图，未看到明显梯度泄漏。
./reassess/r_3.md:4100:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:155:代码主路径基本可信，尤其 `iso_trunk_recce=0 / rec_only=True` 的梯度隔离没有看到实质性泄漏。但结果解释需要收紧：AG-ReID.v2 final 日志不支持“net gain ≈ +1.5 / 等于融合增益”，部分文档还把 ep50 当成 final，且 seed2 / 3090 AIRL 日志不完整。
./reassess/r_3.md:4101:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:169:  `full_map.detach()` 的位置足以切断梯度，但它发生在 f_full Swin late stages 和 `layer4` 已经执行之后。也就是说 degraded `rec_only` pass 仍计算了 f_full 激活并消耗随机层 RNG；只是这些结果不进 loss、不更新 f_full BNNeck/权重。这是注释“cheap / no f_full grad graph built”的夸大，不是梯度泄漏。
./reassess/r_3.md:4102:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:194:代码主路径基本可信，BN/Norm 污染 lens 通过；但结果解释需要收紧。AG-ReID.v2 的同机 final log 不支持“AIRL net gain ≈ +1.5”，只支持“同 run FUSE−full 为正”。
./reassess/r_3.md:4103:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:233:- 注意：非 iso 的 `--airl_dualbranch` degraded pass 是整模型 forward，会让 f_full BNNeck 看到 degraded ground；但本次最终配置是 `--airl_dualbranch_iso`，该污染不存在。
./reassess/r_3.md:4104:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:259:- f_full 初始化参数顺序基本没被 rec 分支污染：f_full BN/classifier 在 rec head 前初始化。
./reassess/r_3.md:4132:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_1.md:7161:../monitor.md:255:- ⚠️ **核心 kill-switch 隐忧**: AIRL 双分支论点 = f_rec(resolution-degradation consistency 训)学到**互补**的 resolvability-aware 表示, 融合后 ≥+1.0。ep10 两头**坍缩成同一表示**, 无互补证据可融。oracle 预言 +1.46, 实测 +0.06。
./reassess/r_3.md:4138:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_1.md:7276:../new_angle_AIRL.md:169:2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一致(同一 degrade+forward 原语), 刻意保持对齐以保证消融诚实; 是否有实质影响由 kill-switch #4 训练结果裁决, 非 bug。若要彻底隔离需加 rec-only forward(改训练行为, 当前 lab-3090 正在跑此代码, 不动)。
./reassess/r_3.md:4204:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp170/monitor.md:85:**结论**: PGMPOA 确认中性。上半身遮挡既不帮忙也不伤害。
./reassess/r_3.md:4205:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp170/monitor.md:99:**结论**: PGMPOA mAP 略低但 R1 更高（+0.7）。上半身遮挡增强可能改善了 top-1 匹配鲁棒性，但牺牲了整体排序（mAP）。总体中性偏微正（R1 角度）。
./reassess/r_3.md:4249:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_4.md:16635:../../decisions.md:4603:**决策**: **纯 CLIP 文本部位路线=死路**(语义冗余)。step2 的新 CLIP 接法必须带 global 没有的真信息(CLIP 视觉特征/遮挡推理/ID 级原型)。理想判据(用户定):baseline 58 → +CLIP 59(CLIP 单独过坎)→ +pose-CLIP 60;现状是 CLIP 单独那步=0。
./reassess/r_3.md:4250:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_4.md:16639:../../decisions.md:4669:### [2026-06-23] 决策 #97: AIRL fusion 零训练 oracle kill-switch = PASS(上完整 resolvability 双分支)
./reassess/r_3.md:4254:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_4.md:16907:../../decisions.md:4669:### [2026-06-23] 决策 #97: AIRL fusion 零训练 oracle kill-switch = PASS(上完整 resolvability 双分支)
./reassess/r_3.md:4328:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_5.md:9912:../new_angle_AIRL.md:169:2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一致(同一 degrade+forward 原语), 刻意保持对齐以保证消融诚实; 是否有实质影响由 kill-switch #4 训练结果裁决, 非 bug。若要彻底隔离需加 rec-only forward(改训练行为, 当前 lab-3090 正在跑此代码, 不动)。
./reassess/r_3.md:4334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_5.md:10169:../../decisions.md:4669:### [2026-06-23] 决策 #97: AIRL fusion 零训练 oracle kill-switch = PASS(上完整 resolvability 双分支)
./reassess/r_3.md:4342:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_5.md:10330:../../../experiments/decisions.md:4669:### [2026-06-23] 决策 #97: AIRL fusion 零训练 oracle kill-switch = PASS(上完整 resolvability 双分支)
./reassess/r_3.md:4362:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:7:- **真没试过的点**：遮挡物 token 语言建模(FLaN-Net)、文本-prompt masking 当遮挡代理(RMPSNet DMPA)、**特征级下半身优先 token 擦除**(RMPSNet RPE，我们只有图像级 PLBOA→ 可试 exp332)。
./reassess/r_3.md:4364:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:22:- **判别性-互补性张力**（FM-import）+ **exp109 三堵墙** + **frozen kill-switch 系统性骗人**
./reassess/r_3.md:4369:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:50:2. **burstiness 抑制（exp329，搬 VLAD-BuFF/face-set）→ KILL**：frozen DINO 上前提成立（遮挡更冗余 +0.0206），但**训练好的弱 baseline 上双判据全 KILL**（burst−uniform −0.29/−0.25，遮挡反而更不冗余）。**meta-finding：ReID 训练已吸收遮挡-burstiness，连弱 baseline 也吸收 → frozen kill-switch 会骗人。**
./reassess/r_3.md:4370:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:51:3. **compositional 组合泛化 + group-DRO（exp330，搬 Sagawa）→ NO-GO**：(遮挡物类×身体部位) held-out 组合，赌模型学 occluder 捷径会崩。**ERM held-out vs seen GAP=+0.10≈0 → 模型本就组合鲁棒**（不学 occluder 捷径）。双审+smoke 全过、单变量干净，kill-switch 便宜判死。
./reassess/r_3.md:4372:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:63:3. **两轮真调研（联网 + 对抗验证，30+ agent / 300万 token）**：现有 occluded 上 8 候选 **0 过审**（每个有真实顶会先例 + 撞墙，有据可查）；TBPS + 亲缘任务 7 候选**活 1 个 = PartNC**（用 pose 可见性区分"文本-图对不上=遮挡 vs =标注错"，复用我们 part-MaxSim + 遮挡老本）。**诚实：PartNC 非稳赢，是探索性赌注，但有干净的 2-3 天廉价 kill-switch。**
./reassess/r_3.md:4415:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/oracle_summary.json:15:  "verdict": "STOP-LOSS: DINO\u2295Swin carries no independent info (P_dino_only=0.20% <2% OR oracle gain +0.12 <+1mAP). Kill the whole 'DINO completes Swin' family."
./reassess/r_3.md:4426:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/monitor.md:40:- 真正的 kill 证据是 **per-query oracle 上界**：即便允许每个 query 取 Swin/DINO 中更好的那个 AP，
./reassess/r_3.md:4477:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp334_smpl_geom/monitor.md:38:- **结论**：SMPL 几何 body-pool **两头都伤**——test 特征伤（location≠visibility，池进遮挡物 token）+ 训练正则也伤（噪声梯度损 backbone）。**location≠visibility 经验证实**。
./reassess/r_3.md:4481:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:14:"transfer_to_occluded": "Adapt the VCS \"corrupt-a-parsed-region + feature-consistency loss\" trick to occlusion: use human parsing (or an occluder mask) to identify non-target/occluded regions, synthesize occlusion by replacing those pixels with random pixels sampled from a per-batch pool, then add an L2/MSE alignment loss pulling the corrupted-image feature toward the clean-image feature. This is a training-only, self-supervised occlusion-augmentation regularizer that teaches the backbone to produce features invariant to corrupted/missing regions, without needing extra inference modules or pose at test time. The batch-pool pixel-shuffle (rather than zero/black masking) is the transferable detail — it injects realistic distractor texture instead of constant fills, which better mimics real occluders.",
./reassess/r_3.md:4482:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:21:"mechanism": "TTPM is a ViT + HRNet pipeline with three parts and no occlusion-sample augmentation. A Multi-patch Feature Encoder encodes intra-patch and inter-patch features (re-grouping patches at 2x scale with the cls token). Pose-Patch Mapping (PPM) builds a fine-grained, per-patch positional correspondence between M pose keypoints and N patches via a learned similarity S = gamma*D_M - eta*S_M that combines an affine-extended Mahalanobis distance and affine-extended cosine similarity, gates out low-confidence keypoints by a heatmap-confidence threshold, and selects the argmax-similarity patch per keypoint, thereby down-weighting occluded patches without coarse global pose mapping. A Texture-Aware Decoder (context-extraction MHSA layers + texture-extraction MHCA layers using pose-filtered features as queries) leverages texture distinctiveness to separate the target from non-target pedestrian occluders. An auxiliary pose loss penalizes high cosine similarity between human-part and non-human (background) features.",
./reassess/r_3.md:4483:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:22:"transfer_to_occluded": "Two directly usable ideas for occluded ReID. (1) The pose loss L_p that explicitly minimizes cosine similarity between body-part features and non-human/background features — a cheap regularizer to suppress occluder/background leakage into the identity embedding, applicable to any pose- or part-based occluded model. (2) The texture-distinctiveness decoder channel aimed specifically at non-target PEDESTRIAN occlusion: when the occluder is itself a person, pose/structure cues fail (same skeleton topology), so using texture appearance distinctiveness (via pose-filtered queries cross-attending to contextual features) to push apart target vs distractor-person is a concrete mechanism for the under-addressed person-on-person occlusion case. Also transferable: replacing coarse pose-to-global mapping with a per-keypoint-to-patch confidence-gated argmax correspondence (Mahalanobis+cosine affine similarity) to localize visible regions.",
./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_3.md:4491:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:86:"transfer_to_occluded": "Apply the text-level orthogonal-projection disentanglement to occlusion: learn a dual prompt for the OCCLUDER/distractor (e.g. \"a photo of the [X]...[X] occluder\") alongside the ID prompt, then orthogonalize the occluder text feature out of the ID text feature (F_ort = F_id - proj of F_occ onto F_id) and use the purged ID text feature to guide the visual encoder via the non-local SIM. Because the non-local operation aggregates from spatially distant but semantically relevant regions, it is well suited to occluded ReID where visible body parts are scattered far from the occluded/corrupted region — letting the model pull ID evidence from non-adjacent visible patches rather than local self-attention that gets dominated by the occluder. The orthogonalization gives a clean way to subtract a known nuisance direction (occluder appearance) without an auxiliary parser.",
./reassess/r_3.md:4492:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:88:"novel_component": "Disentangling a nuisance factor by vector orthogonal projection in the TEXT embedding space: project the nuisance (clothing) text feature onto the ID text direction and subtract it, purging the nuisance-aligned component of ID features without any auxiliary visual model — a cheap, parser-free disentanglement that generalizes to any named confounder (including occluders)."
./reassess/r_3.md:4493:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:102:"transfer_to_occluded": "Use real out-of-domain vehicle (or generic object) crops as the occluder source for copy-paste augmentation instead of synthetic gray patches or random erasing, and inject them with a controlled height-ratio so the paste lands on the lower body to mimic real vehicle-induced occlusion. This produces texture-and-color-realistic distractor occluders that better match deployment-time occlusions than random-erasing, and the epoch-wise alternation between clean and occluder-domain data acts as a regularizer pushing the model toward occlusion-invariant features — both are drop-in, annotation-free, and backbone-agnostic for any occluded ReID pipeline.",
./reassess/r_3.md:4500:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:142:"transfer_to_occluded": "Use an off-the-shelf 3D body reconstruction (SMPL/HMR) of the pedestrian to generate geometric reference points, then drive deformable cross-attention so the image-feature sampling locations are anchored at projected body-keypoint/mesh positions. For occluded ReID this gives a body-structure prior independent of pixel evidence: even when a region is occluded, the reconstructed mesh still provides a body-anchored query, steering attention onto plausible person regions and away from distractor occluders, and the pose/shape-derived reference points act as a soft visibility/structure guide. The SSTI idea of an appearance-independent shape descriptor (from β only) is also a complementary, occlusion-robust feature since estimated body shape is less corrupted by occluders than RGB texture.",
./reassess/r_3.md:4529:/Users/abdslm/Desktop/SOLIDER-REID/experiments/results.md:336:**结论**: PSG 在所有数据集和 backbone 上均有效。在遮挡数据集上增益更大。
./reassess/r_3.md:4544:/Users/abdslm/Desktop/SOLIDER-REID/experiments/results.md:1667:- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。
./reassess/r_3.md:4551:/Users/abdslm/Desktop/SOLIDER-REID/experiments/results.md:1755:| exp354 PC-SOR | pose+CLIP文本 token归属(20-codex首推) | kill-switch FAILED | CLIP文本定位不了遮挡物/分不清目标vs任意人 |
./reassess/r_3.md:4649:/Users/abdslm/Desktop/SOLIDER-REID/experiments/publishable_angles_synthesis.md:40:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:4686:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:4:> 战略：搬外部范式（CLIP-ReID/Pose2ID 打法），冻结 kill-switch 先验 → 赢家训头 → 破天花板。已死：MLLM-reasoner（姿态提示无效）。已证伪：exp109 内部机制（三堵墙）。
./reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
./reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
./reassess/r_3.md:4708:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:95:**判定：STOP-LOSS — DINO⊕Swin 无独立信息，整条"DINO 补 Swin"家族止损。**（P_dino_only 0.20%<2% 且 oracle gain +0.12<+1mAP，两条 kill 条件双中。）
./reassess/r_3.md:4712:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:106:重遮挡子集(989 query)：top-10 Jaccard **0.062** · **P_dino_only 0.20%**(2/989) · Swin-only **72.57** · DINO-only **8.65** · **oracle 上界 72.69(+0.12)**。两个 kill 条件全中。
./reassess/r_3.md:4719:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:133:- **直接坐实 oracle agent 的判断**：「frozen-source 换源(RADIO/SAM2/DIFT/DINOv3)价值大降」现在有**全量实证**——换更新 DINO（registers）只 +0.29，换 SD 范式（DIFT）反而 −1.13。**换源都不破 frozen 天花板，瓶颈在 frozen 本身**。DINO 线唯一活口仍是 exp324d LoRA（解冻），与 oracle 结论收敛。
./reassess/r_3.md:4720:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:138:- **hyy 探针(full Occ-Duke)**：DINOv2-registers **2.15**(+0.29，没破)、**DIFT(SD-v1.5) 0.73(−1.13，更差)**。SD 特征强于 category 对应但弱于 instance ID 判别(SD-DINO 文献一致)。**结论：天花板瓶颈是"冻结"本身，不是模型新旧** → 独立佐证 oracle。换源(RADIO/SAM2/DIFT/DINOv3)全砍。
./reassess/r_3.md:4726:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:185:- **这推翻了 oracle/换源探针的"FM 对 occluded ReID 无独立信息"悲观结论的一半**：那些结论是**冻结**条件下成立（换更新 DINO +0.29、SD/DIFT −1.13、oracle 无独立信息）。**一旦允许 LoRA 解冻，瓶颈消失** → 证实瓶颈是"冻结"本身，不是 DINO 表征结构、不是模型新旧。
./reassess/r_3.md:4735:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:252:**结论：FM-import 方法猎彻底关闭（MLLM/frozen/换源/LoRA-standalone/LoRA-fusion 全负，各有机制）。** large(15h dead-end) 已 kill。**夜间 useful 产出 = `fm_occluded_reid_study.md` 诊断研究（含张力洞察 + ×4 finding + 可复用 oracle 工具）**——一个诚实、证据扎实、对领域有用的 analysis 贡献（劝退+量化+解释"搬通用 FM 赢 occluded ReID"这个诱人方向）。
./reassess/r_3.md:4737:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:283:**判据**：λ=1 vs λ=0 看 (1) decorr-DINO 单分支 mAP 是否被解相关拖垮；(2) top-10 Jaccard 是否下降（张力是否被打破）；(3) fusion(decorr-DINO⊕Swin) 重遮挡/全部是否真超 Swin。**无论成败都有价值**：成→真 method；败→把张力从"观察"升级为"显式施压也打不破"的强诊断结论。
./reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
./reassess/r_3.md:4746:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:357:### [λ=1 e30 oracle 数已captured，verdict 等 matched λ=0/λ=2 e30]
./reassess/r_3.md:4756:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:389:**唯一幸存 PartNC(Part-level Noisy-Correspondence)**：用部位粒度 image-text 相似度估每个身体部位 clean/noisy 置信(RDE 的 CCD 从 pair 级下沉到部位级)；**真正空白=用 pose-visibility 先验区分"相似度低=遮挡(结构化)"vs"=标注错(随机)"**(RDE/DURA/GA-DMS 把 noise 当随机，表达不出)。复用我们 part-MaxSim 逻辑(einsum bkc,gpc->bgkp .max)+pose 可见性积累。
./reassess/r_3.md:4769:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:472:- **0-GPU kill-switch 已搭并验证**(`scripts/burstiness_probe.py`, staged on hyy): 复用 TransReID 自己的 make_dataloader/make_model/R1_mAP_eval(mAP 可直接对比训练日志), hook `base.norm` 取全 token 序列(B,129,768), 每图可见 token 集内算 burst 权重 → burst 加权 patch-pool vs **uniform patch-pool**(隔离 burstiness 机制, cls 仅作 context) → 重排比 mAP。**判据: burst−uniform ≥+1.0=真值得全量, <+0.3=kill。** + 诊断(query 遮挡 vs gallery 整体的集内 self-similarity)。
./reassess/r_3.md:4795:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp337_swin_lgpa_nopose/design.md:37:- **step2 启示**：纯 CLIP 文本路线无效（语义冗余于 global）。新 CLIP 接法要么保留 pose 引导，要么换带 global 没有的新信息的 CLIP 信号（CLIP 视觉特征 / 遮挡推理 / ID 原型），不能靠 CLIP 文本部位语义本身。
./reassess/r_3.md:4816:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:59:1. **诊断/负面结论**（劝退 + 量化）：MLLM 不会遮挡推理、冻结 FM 无独立信息、换源无用——省别人踩坑。
./reassess/r_3.md:4869:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp329_burstiness/design.md:66:- **结论**：burstiness 在训练好的 ReID 模型上无 headroom，即便弱 baseline。ReID 训练已隐式吸收遮挡-burstiness 结构。归入"in-domain 特征机制 frozen-promising / trained-absorbed" pattern。
./reassess/r_3.md:4918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp340c_randomtext/claude_review.md:73:- [x] generator-seeded randn 确定性可复现、CPU 构造、不污染全局 RNG（实跑确认）
./reassess/r_3.md:4925:/Users/abdslm/Desktop/SOLIDER-REID/experiments/paper_notes/paper_16_ProFD.md:6:**arXiv 摘要**: 利用 CLIP 视觉-语言模型，通过部位特定的文本 prompt 引导特征解耦，结合混合注意力 Decoder 和自蒸馏策略处理遮挡人员重识别。
./reassess/r_3.md:4950:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/design.md:3:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5042:/Users/abdslm/Desktop/SOLIDER-REID/experiments/paper_materials/story.md:1225:1. 我们想保留的不是“oracle query context”
./reassess/r_3.md:5051:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324/monitor.md:52:- **值得下一步**：上轻量 part-projection 头 / LoRA 把 DINO 特征投到 ReID-judiciable 空间，再全量对比 KPR（design.md kill-switch 命中"重遮挡组超 holistic 且 pose 锚定有效" → exp324b）。
./reassess/r_3.md:5052:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review.log:22:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5070:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review.log:1160:experiments/results.md:1667:- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。
./reassess/r_3.md:5072:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review.log:1170:experiments/exp324b/design.md:3:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5075:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review.log:1184:experiments/exp324b/_codex_review.log:22:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5082:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review.log:1252:experiments/exp324/monitor.md:52:- **值得下一步**：上轻量 part-projection 头 / LoRA 把 DINO 特征投到 ReID-judiciable 空间，再全量对比 KPR（design.md kill-switch 命中"重遮挡组超 holistic 且 pose 锚定有效" → exp324b）。
./reassess/r_3.md:5083:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review.log:1487:     3	> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5104:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:1737:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5112:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2093:     3	> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5120:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2435:./experiments/results.md:1667:- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。
./reassess/r_3.md:5122:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2494:./experiments/exp324b/_codex_review.log:1160:experiments/results.md:1667:- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。
./reassess/r_3.md:5123:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2496:./experiments/exp324b/_codex_review.log:1170:experiments/exp324b/design.md:3:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5126:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2508:./experiments/exp324b/_codex_review.log:1184:experiments/exp324b/_codex_review.log:22:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
./reassess/r_3.md:5130:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2568:./experiments/exp324b/_codex_review.log:1252:experiments/exp324/monitor.md:52:- **值得下一步**：上轻量 part-projection 头 / LoRA 把 DINO 特征投到 ReID-judiciable 空间，再全量对比 KPR（design.md kill-switch 命中"重遮挡组超 holistic 且 pose 锚定有效" → exp324b）。
./reassess/r_3.md:5131:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2647:./experiments/exp324b/claude_review.md:12:不是。exp324 frozen 已通过 kill-switch（重遮挡 pose part-MaxSim 1.86 mAP，×3.4 超整图 0.55，且均匀网格对照只 0.67 几乎不涨 → 涨点来自"姿态锚定"机制本身）。exp324b 是这个正信号的**自然下一步**：在冻结特征上训最小投影头，验证"冻结 FM 特征能否被轻量头拉到可用 ReID 判别空间"。单变量隔离干净（baseline = exp324 frozen 同 pipeline，只多了一个 413K 线性头 + BNNeck + 分类器）。这是范式/参数角度的合法探索，不属于禁止回退方向（不是 visibility 小改、不是 retrieval-side scorer 微变体、不是 test-time trick）。重点应放在实现正确性，下文逐行审。
./reassess/r_3.md:5133:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/_codex_review2.log:2745:./experiments/exp324/monitor.md:52:- **值得下一步**：上轻量 part-projection 头 / LoRA 把 DINO 特征投到 ReID-judiciable 空间，再全量对比 KPR（design.md kill-switch 命中"重遮挡组超 holistic 且 pose 锚定有效" → exp324b）。
./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
./reassess/r_3.md:5143:# GOPL 零训练 kill-switch 设计（2026-06-24, B containment 死后转向; novelty 7/10 存活）
./reassess/r_3.md:5155:## 零训练 kill-switch（先验隐藏变量存在 + 证 SMPL 几何独特）
./reassess/r_3.md:5184:SMPL 缓存 lab-3090 `/root/work/SOLIDER-REID/cache/smpl_geom/`。occluded_duke `/root/work/SOLIDER-REID/data/occluded_duke`。强 ckpt agent 在 `log/occluded_duke/` 找。复用 error_analysis_geom.py / cvpb_containment_killswitch.py 的 extract/per_query_ap 基建。
./reassess/r_3.md:5255:./reviews/deep_28.md:869:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5264:./reviews/deep_28.md:6561:  2225	advantage compared with IRRA when source domain is RSTPReid, which gains a significant improvement on R@1 by
./reassess/r_3.md:5268:./reviews/deep_9.md:512:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5288:./reviews/deep_18.md:745:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5363:./reviews/deep_2.md:432:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5365:./reviews/deep_2.md:874:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:5443:./reviews/deep_27.md:648:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5545:./reviews/deep_16.md:588:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5548:./reviews/deep_16.md:1194:/bin/zsh -lc "pdftotext -f 1 -l 8 'Instant pose extraction based on mask transformer for occluded person re-identification.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|mask|pose|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:5568:./reviews/deep_16.md:2471:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:5691:./reviews/deep_5.md:373:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5741:./reviews/deep_6.md:699:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:5742:./reviews/deep_6.md:757:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:5746:./reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:5762:./cvpb_containment_killswitch_design.md:44:无 ID label 泄漏，eval_dist 升序+same pid&cam junk removal 正确。唯一 Low: C3 用 per-dim mean 非 median(spec 本就写"均值方差"，已改 label 为 view-mean)。
./reassess/r_3.md:5763:./reviews/deep_17.md:398:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:5807:./reviews/deep_17.md:4972:   111	key idea is that our model is trained on a set of source domain based on the unpaired images/texts
./reassess/r_3.md:5823:./reviews/deep_23.md:664:misalignment between the visual and textual features caused by occlusions. These challenges motivate us
./reassess/r_3.md:5927:./reviews/deep_23.md:3311:前两篇已经读到核心：一篇把遮挡解释成真实遮挡分布、文本语义缺失和增强后特征分布不一致三件事；另一篇把退化 ReID 从“单一退化域偏移”改写成“混合退化下的分布偏移加身份拓扑扭曲”。我会补一下被输出截断的中间方法细节，然后继续后四篇。
./reassess/r_3.md:5978:./reviews/deep_23.md:8955:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:5980:./reviews/deep_23.md:9005:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:5995:./reviews/deep_4.md:788:with a occlusion-aware fine-grained textual prompt in the format:”A photo of a [S ∗ ] person with [A∗ ] partially occluded
./reassess/r_3.md:6039:./reviews/deep_4.md:11383:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:6040:./reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:6041:./reviews/deep_4.md:11460:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:6042:./reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:6046:./reviews/deep_11.md:740:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:6129:./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
./reassess/r_3.md:6176:./reviews/deep_20.md:387:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:6180:./reviews/deep_20.md:960:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:6206:./reviews/deep_20.md:2165:    17	and occlusions, by capturing both fine-grained details and broad contextual cues. Additionally,
./reassess/r_3.md:6276:./reviews/deep_20.md:5980:   748	In the context of occluded person re-identification, occlusions increase task entropy because visual cues become incomplete and ambiguous. By injecting structured ‘‘vehicle noise’’ in the form of semantically coherent patterns (e.g., vehicle textures and contours), the
./reassess/r_3.md:6302:./reviews/deep_14.md:1287:161:with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reassess/r_3.md:6351:./reviews/deep_14.md:3974:   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reassess/r_3.md:6361:./reviews/deep_14.md:5763:161:   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reassess/r_3.md:6400:./reviews/deep_3.md:2932:    57	However, in the context of occlusion ReID tasks, these models face
./reassess/r_3.md:6433:./reviews/deep_15.md:2829:texture resources in synthetic datasets and enhance both
./reassess/r_3.md:6435:./reviews/deep_15.md:2853:the source image and the synthetic image, aiming to produce high-quality texture images. Zhang et al. [19] used
./reassess/r_3.md:6465:./reviews/deep_15.md:9575:   121	texture resources in synthetic datasets and enhance both
./reassess/r_3.md:6467:./reviews/deep_15.md:9599:   145	the source image and the synthetic image, aiming to produce high-quality texture images. Zhang et al. [19] used
./reassess/r_3.md:6473:./reviews/deep_15.md:12068:   121	texture resources in synthetic datasets and enhance both
./reassess/r_3.md:6486:./reviews/deep_15.md:13141:  1149	a massive visual resource library for images of a particular pedestrian, based on a textual query. However, existing
./reassess/r_3.md:6493:./pivot/p_3.md:1994:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:6502:./pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
./reassess/r_3.md:6503:./pivot/p_3.md:2026:reid-paper 项目 D4 诊断**已确认 donor-leakage 人际混淆在强主干仍持续(≈3×物体干扰)**——kill-switch 其实已经过了。衍生"donor-leakage 反事实一致性"方法 6-7/10、待探针。**库读证明: 你这条自己的活口, 正好是整个领域的空白。**
./reassess/r_3.md:6507:./pivot/p_3.md:2045:**领域(167篇)+ 你自己的 D4 诊断 + 获胜配方 三方收敛到同一个角度: 同框入侵身份泄漏的 source-separation(非补全)。** 这是 reid-paper 项目最强活口, 也是全领域空白, 且自带打 wall-2 的对照 + 一晚的 kill-switch。下一步: 核对"主要人物归属"当年判死理由 → 不撞则跑探针 → 建 donor-leakage 反事实一致性方法。
./reassess/r_3.md:6512:./pivot/p_3.md:2185:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:6516:./pivot/p_3.md:2219:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:6526:./reviews/deep_1.md:1055:/bin/zsh -lc 'pdftotext -f 1 -l 8 "A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:6545:./reviews/deep_1.md:4495:/bin/zsh -lc 'pdftotext "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n -C 2 "Ablation|ablation|visual|Visualization|t-SNE|CMF|SSTI|3D|shape|occlusion|misalignment"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:6565:./reviews/deep_1.md:7603:/bin/zsh -lc "pdftotext -f 1 -l 8 \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | sed -n '1,675p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:6593:./reviews/deep_1.md:7740:generates semantically relevant occlusion patches with semantic information of object pedestrians and further realizes occlusion simulation based on their position, shape, and context to generate more
./reassess/r_3.md:6650:./reviews/deep_1.md:8292:/bin/zsh -lc "pdftotext -layout -f 7 -l 11 \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | sed -n '1,500p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:6674:./reviews/deep_1.md:8665:/bin/zsh -lc "pdftotext \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | rg -n -C 3 \"Ablation|ablation|SGOS|LFSE|cross|offset|occlusion|semantic|visual|Fig\\. 5|heatmap|t-SNE|Grad|effectiveness|component\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:6702:./reviews/deep_1.md:8776:136:generates semantically relevant occlusion patches with semantic information of object pedestrians and further realizes occlusion simulation based on their position, shape, and context to generate more
./reassess/r_3.md:6810:./analyses/deep_23.txt:6:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:6832:./reviews/deep_10.md:4210:in lighting, viewpoints, and camera sources. Fourth, textto-image methods [4, 43, 44] leverage textual descriptions
./reassess/r_3.md:7037:./reviews/deep_21.md:6604:   126	color and texture, while in infrared images, occlusions often
./reassess/r_3.md:7040:./reviews/deep_21.md:6634:   156	occlusion from a completion perspective rather than a discarding one. Specifically, we propose to fully exploit both intramodality contextual cues and inter-modality complementary
./reassess/r_3.md:7046:./reviews/deep_21.md:6704:   226	information with visual features to enhance contextual representation under occlusion. Chen et al. [17] introduced a
./reassess/r_3.md:7091:./analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7116:./analyses/ALL_ANALYSES.txt:1016:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:7117:./analyses/ALL_ANALYSES.txt:1115:我主要依据本地 PDF 全文方法和消融页阅读，另外核对了 ScRL 和 FedSTIL 的 arXiv 页面，以及弱监督文本 ReID 相关背景 CPCL 页面：ScRL arXiv([arxiv.org](https://arxiv.org/abs/2310.17952?utm_source=openai))、FedSTIL arXiv([arxiv.org](https://arxiv.org/abs/2207.11759?utm_source=openai))、CPCL arXiv([arxiv.org](https://arxiv.org/abs/2401.10011?utm_source=openai))。
./reassess/r_3.md:7119:./analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7128:./analyses/ALL_ANALYSES.txt:1524:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:7129:./analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:7134:./reviews/deep_24.md:403:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7138:./reviews/deep_24.md:976:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:7172:./reviews/deep_24.md:17079:我主要依据本地 PDF 全文方法和消融页阅读，另外核对了 ScRL 和 FedSTIL 的 arXiv 页面，以及弱监督文本 ReID 相关背景 CPCL 页面：ScRL arXiv([arxiv.org](https://arxiv.org/abs/2310.17952?utm_source=openai))、FedSTIL arXiv([arxiv.org](https://arxiv.org/abs/2207.11759?utm_source=openai))、CPCL arXiv([arxiv.org](https://arxiv.org/abs/2401.10011?utm_source=openai))。
./reassess/r_3.md:7173:./reviews/deep_24.md:17136:我主要依据本地 PDF 全文方法和消融页阅读，另外核对了 ScRL 和 FedSTIL 的 arXiv 页面，以及弱监督文本 ReID 相关背景 CPCL 页面：ScRL arXiv([arxiv.org](https://arxiv.org/abs/2310.17952?utm_source=openai))、FedSTIL arXiv([arxiv.org](https://arxiv.org/abs/2207.11759?utm_source=openai))、CPCL arXiv([arxiv.org](https://arxiv.org/abs/2401.10011?utm_source=openai))。
./reassess/r_3.md:7174:./analyses/deep_4.txt:53:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:7175:./analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:7182:./reviews/deep_25.md:2048:more computational resources and imposes stricter data quality requirements. In this context, Ahmad et al. [5] pioneered person matching using
./reassess/r_3.md:7197:./reviews/deep_25.md:6745:pedestrian occlusion, a Texture-Aware Decoder is introduced to perceive texture features and leverage their
./reassess/r_3.md:7227:./reviews/deep_25.md:6852:non-targets. As the appearance of target and non-target pedestrians often exhibits notable texture variations, it is more feasible to leverage
./reassess/r_3.md:7281:./reviews/deep_25.md:8155:texture-aware encoder (left column), TTPM accurately re-identiﬁes the
./reassess/r_3.md:7284:./reviews/deep_25.md:8160:textures. This result indicates that the proposed TTPM still has limitations in relying on texture features. While these features enable accurate
./reassess/r_3.md:7298:./reviews/deep_25.md:12347:TTPM 的页面排版导致方法文本顺序有些乱，刚才截取漏了 MFE 的前半段。我会再补一次该论文从 3.1 到 3.3 的开头，避免把 pose-patch mapping 的逻辑链断掉。
./reassess/r_3.md:7315:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7318:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7326:./pivot/clean/p_3.txt:29:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:7328:./analyses/deep_24.txt:55:我主要依据本地 PDF 全文方法和消融页阅读，另外核对了 ScRL 和 FedSTIL 的 arXiv 页面，以及弱监督文本 ReID 相关背景 CPCL 页面：ScRL arXiv([arxiv.org](https://arxiv.org/abs/2310.17952?utm_source=openai))、FedSTIL arXiv([arxiv.org](https://arxiv.org/abs/2207.11759?utm_source=openai))、CPCL arXiv([arxiv.org](https://arxiv.org/abs/2401.10011?utm_source=openai))。
./reassess/r_3.md:7347:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:7367:./validate/v_2.md:1307:./reviews/deep_23.md:664:misalignment between the visual and textual features caused by occlusions. These challenges motivate us
./reassess/r_3.md:7405:./validate/v_2.md:2455:./reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:7430:./validate/v_2.md:5109:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7431:./validate/v_2.md:5121:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7433:./validate/v_2.md:5173:./analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7442:./validate/v_2.md:5932:./analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7460:./validate/v_2.md:6542:./reviews/deep_1.md:4495:/bin/zsh -lc 'pdftotext "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n -C 2 "Ablation|ablation|visual|Visualization|t-SNE|CMF|SSTI|3D|shape|occlusion|misalignment"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:7478:./validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7479:./validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7510:./validate/v_2.md:7883:validate/v_2.md:2455:./reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:7544:./validate/v_2.md:9360:reviews/deep_21.md:6604:   126	color and texture, while in infrared images, occlusions often
./reassess/r_3.md:7548:./validate/v_2.md:9697:reviews/deep_10.md:4210:in lighting, viewpoints, and camera sources. Fourth, textto-image methods [4, 43, 44] leverage textual descriptions
./reassess/r_3.md:7558:./validate/v_2.md:10785:reviews/deep_15.md:2853:the source image and the synthetic image, aiming to produce high-quality texture images. Zhang et al. [19] used
./reassess/r_3.md:7559:./validate/v_2.md:10838:reviews/deep_15.md:9599:   145	the source image and the synthetic image, aiming to produce high-quality texture images. Zhang et al. [19] used
./reassess/r_3.md:7560:./validate/v_2.md:10892:reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:7567:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7568:./validate/v_2.md:12861:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7570:./validate/v_2.md:12865:validate/v_3.md:4053:../litreview/reviews/lit_15.md:4249:cheap kill-switch：冻结 SOLIDER，在航拍地面数据上按高度、分辨率、视角、遮挡分桶，计算同身份邻域保持率和跨桶检索掉点。若只是整体中心偏移，简单白化或域对齐就能恢复，那这个方向不够强。
./reassess/r_3.md:7571:./validate/v_2.md:12867:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7573:./validate/v_2.md:12871:validate/v_3.md:4069:../litreview/reviews/lit_15.md:4305:cheap kill-switch：冻结 SOLIDER，在航拍地面数据上按高度、分辨率、视角、遮挡分桶，计算同身份邻域保持率和跨桶检索掉点。若只是整体中心偏移，简单白化或域对齐就能恢复，那这个方向不够强。
./reassess/r_3.md:7576:./validate/v_2.md:13709:validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7577:./validate/v_2.md:13710:validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7579:./validate/v_2.md:14462:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:7583:./validate/v_2.md:14516:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:7591:./validate/v_3.md:760:./reviews/deep_23.md:8955:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:7593:./validate/v_3.md:781:./reviews/deep_23.md:9005:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:7614:./validate/v_3.md:1887:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7617:./validate/v_3.md:1906:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7620:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7628:./validate/v_3.md:2902:../litreview2/analyses/ALL_ANALYSES.txt:1016:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:7630:./validate/v_3.md:2945:../litreview2/analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:7635:./validate/v_3.md:3018:../litreview2/analyses/ALL_ANALYSES.txt:1524:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:7636:./validate/v_3.md:3019:../litreview2/analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:7639:./validate/v_3.md:4044:../litreview/reviews/lit_15.md:4226:创新类型是新设定细化加工程组合。gap 是遮挡 ReID 里随机擦除和真实遮挡分布不一致，且 CLIP prompt 对缺失语义不稳。它用下半身优先擦除、双 masked prompt、多分支分布对齐来补。story 是“训练遮挡必须像真实遮挡，文本 prompt 也要学会缺词时仍保持身份语义”。
./reassess/r_3.md:7640:./validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7642:./validate/v_3.md:4053:../litreview/reviews/lit_15.md:4249:cheap kill-switch：冻结 SOLIDER，在航拍地面数据上按高度、分辨率、视角、遮挡分桶，计算同身份邻域保持率和跨桶检索掉点。若只是整体中心偏移，简单白化或域对齐就能恢复，那这个方向不够强。
./reassess/r_3.md:7643:./validate/v_3.md:4060:../litreview/reviews/lit_15.md:4282:创新类型是新设定细化加工程组合。gap 是遮挡 ReID 里随机擦除和真实遮挡分布不一致，且 CLIP prompt 对缺失语义不稳。它用下半身优先擦除、双 masked prompt、多分支分布对齐来补。story 是“训练遮挡必须像真实遮挡，文本 prompt 也要学会缺词时仍保持身份语义”。
./reassess/r_3.md:7644:./validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7646:./validate/v_3.md:4069:../litreview/reviews/lit_15.md:4305:cheap kill-switch：冻结 SOLIDER，在航拍地面数据上按高度、分辨率、视角、遮挡分桶，计算同身份邻域保持率和跨桶检索掉点。若只是整体中心偏移，简单白化或域对齐就能恢复，那这个方向不够强。
./reassess/r_3.md:7691:./validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7693:./validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7695:./validate/v_3.md:8721:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7696:./validate/v_3.md:8725:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7698:./validate/v_3.md:8984:./validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7699:./validate/v_3.md:8987:./validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7700:./validate/v_3.md:9203:../litreview2/validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7701:./validate/v_3.md:9206:../litreview2/validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7702:./validate/v_3.md:9939:./validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7703:./validate/v_3.md:9940:./validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7704:./validate/v_3.md:10335:./validate/v_2.md:12861:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7705:./validate/v_3.md:10337:./validate/v_2.md:12867:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7706:./validate/v_3.md:10419:./validate/v_2.md:13709:validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7707:./validate/v_3.md:10420:./validate/v_2.md:13710:validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7709:./validate/v_3.md:10904:../litreview2/validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7710:./validate/v_3.md:10905:../litreview2/validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7711:./validate/v_3.md:11300:../litreview2/validate/v_2.md:12861:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7712:./validate/v_3.md:11302:../litreview2/validate/v_2.md:12867:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7713:./validate/v_3.md:11384:../litreview2/validate/v_2.md:13709:validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7714:./validate/v_3.md:11385:../litreview2/validate/v_2.md:13710:validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7716:./validate/v_3.md:14184:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7718:./validate/v_3.md:15168:../litreview2/validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7719:./validate/v_3.md:15494:../litreview2/validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7720:./validate/v_3.md:16098:../litreview2/validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:7723:./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_3.md:7730:./reassess/r_3.md:1785:主文档确认了一个关键背景：遮挡方向已有大量“补全、可见性、结构关系、masking、metric alignment”负结果，且 exp109 一类结论确实在说可实现的遮挡处理很容易退化成 identity-conditioned oracle 不可达。现在我会专门检索本地 GOPL、source-separation、occluder-gate 相关材料，再做外部 novelty 对照。
./reassess/r_3.md:7734:./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
./reassess/r_3.md:7738:./reassess/r_3.md:1983:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/cvpb_containment_killswitch_design.md:44:无 ID label 泄漏，eval_dist 升序+same pid&cam junk removal 正确。唯一 Low: C3 用 per-dim mean 非 median(spec 本就写"均值方差"，已改 label 为 view-mean)。
./reassess/r_3.md:7743:./reassess/r_3.md:2034:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:869:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7747:./reassess/r_3.md:2056:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:512:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7761:./reassess/r_3.md:2167:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:648:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7766:./reassess/r_3.md:2212:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:1994:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:7775:./reassess/r_3.md:2223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
./reassess/r_3.md:7776:./reassess/r_3.md:2224:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2026:reid-paper 项目 D4 诊断**已确认 donor-leakage 人际混淆在强主干仍持续(≈3×物体干扰)**——kill-switch 其实已经过了。衍生"donor-leakage 反事实一致性"方法 6-7/10、待探针。**库读证明: 你这条自己的活口, 正好是整个领域的空白。**
./reassess/r_3.md:7780:./reassess/r_3.md:2231:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2045:**领域(167篇)+ 你自己的 D4 诊断 + 获胜配方 三方收敛到同一个角度: 同框入侵身份泄漏的 source-separation(非补全)。** 这是 reid-paper 项目最强活口, 也是全领域空白, 且自带打 wall-2 的对照 + 一晚的 kill-switch。下一步: 核对"主要人物归属"当年判死理由 → 不撞则跑探针 → 建 donor-leakage 反事实一致性方法。
./reassess/r_3.md:7785:./reassess/r_3.md:2244:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2185:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:7789:./reassess/r_3.md:2250:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2219:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:7792:./reassess/r_3.md:2278:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:588:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7808:./reassess/r_3.md:2352:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/p_3.txt:29:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
./reassess/r_3.md:7815:./reassess/r_3.md:2368:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp330_compositional_dro/design.md:33:- **前置 GAP 检查（先跑 ERM）**：ERM 自己 held-out mean mAP 是否显著 < seen mean（存在组合 GAP）。若 ERM 无 GAP（held≈seen）→ 无 occluder-class 捷径可利用 → 整 bet NO-GO（合理早 kill，省 DRO）。有 GAP 才看 DRO 能否合上。
./reassess/r_3.md:7823:./reassess/r_3.md:2417:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:745:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7829:./reassess/r_3.md:2469:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:403:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7833:./reassess/r_3.md:2499:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:976:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:7838:./reassess/r_3.md:2594:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:699:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:7839:./reassess/r_3.md:2595:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:757:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7843:./reassess/r_3.md:2604:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:432:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7845:./reassess/r_3.md:2631:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:874:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:7856:./reassess/r_3.md:2748:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:398:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:7861:./reassess/r_3.md:2848:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:835:./analyses/deep_4.txt:53:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:7862:./reassess/r_3.md:2849:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:836:./analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:7866:./reassess/r_3.md:2889:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1061:./analyses/ALL_ANALYSES.txt:1524:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:7867:./reassess/r_3.md:2890:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1062:./analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:7868:./reassess/r_3.md:2960:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1493:./reviews/deep_4.md:11383:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:7869:./reassess/r_3.md:2961:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1494:./reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:7870:./reassess/r_3.md:2965:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1500:./reviews/deep_4.md:11460:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:7871:./reassess/r_3.md:2966:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1501:./reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:7874:./reassess/r_3.md:3044:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2277:./validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7875:./reassess/r_3.md:3045:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2278:./validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:7885:./reassess/r_3.md:3281:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5136:pivot/p_3.md:1994:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:7891:./reassess/r_3.md:3309:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_7.md:8446:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp330_compositional_dro/design.md:33:- **前置 GAP 检查（先跑 ERM）**：ERM 自己 held-out mean mAP 是否显著 < seen mean（存在组合 GAP）。若 ERM 无 GAP（held≈seen）→ 无 occluder-class 捷径可利用 → 整 bet NO-GO（合理早 kill，省 DRO）。有 GAP 才看 DRO 能否合上。
./reassess/r_3.md:7897:./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_3.md:7901:./reassess/r_3.md:3351:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:1785:主文档确认了一个关键背景：遮挡方向已有大量“补全、可见性、结构关系、masking、metric alignment”负结果，且 exp109 一类结论确实在说可实现的遮挡处理很容易退化成 identity-conditioned oracle 不可达。现在我会专门检索本地 GOPL、source-separation、occluder-gate 相关材料，再做外部 novelty 对照。
./reassess/r_3.md:7906:./reassess/r_3.md:3404:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/new_angle_AIRL.md:169:2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一致(同一 degrade+forward 原语), 刻意保持对齐以保证消融诚实; 是否有实质影响由 kill-switch #4 训练结果裁决, 非 bug。若要彻底隔离需加 rec-only forward(改训练行为, 当前 lab-3090 正在跑此代码, 不动)。
./reassess/r_3.md:7933:./reassess/r_3.md:3910:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:312:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:7947:./reassess/r_3.md:4045:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:7:- **机制**：CLIP-based（冻结 ViT-B/16 图像+文本编码器）。每张遮挡图做 textual inversion 构造细粒度 prompt `"A photo of a [S*] person with [A*] partially occluded by [O*]"`：S*=主体身份 token、A*=可见属性 token（衣着/配饰，learnable queries+cross-attn）、**O*=遮挡物 token（显式在语言空间建模遮挡物本身）**。cross-attn 对齐 prompt 与可见区域 + 熵不确定性加权融合 {视觉,文本,cross-attn}。
./reassess/r_3.md:7950:./reassess/r_3.md:4062:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:78:| SCI | CLIP 文本正交投影减 occluder 方向 | = **FM-import CLIP** + 特征对齐 |
./reassess/r_3.md:7953:./reassess/r_3.md:4096:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:4:代码层面我没看到 AIRL eval/fuse 的关键实现错误，也没看到 FUSE 使用 query/gallery label 的泄漏。  
./reassess/r_3.md:7954:./reassess/r_3.md:4097:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:10:  训练中用 query/gallery eval 结果保存 `model_best`，若论文报告 best 而不是 final，会构成 test-set model selection 泄漏。当前 AG-ReID.v2 4090 主日志 best=final epoch60，所以这次主数未被实际污染。
./reassess/r_3.md:7955:./reassess/r_3.md:4098:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:42:- 主要风险不是代码泄漏，而是实验口径：不要报告 test-best；AG-ReID.v2 当前只能 claim “融合互补”，不能 claim “净超 baseline”。
./reassess/r_3.md:7956:./reassess/r_3.md:4099:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:149:degradation 实现是：只对 ground 样本做 normalized tensor 空间的 bilinear downsample 再 upsample，`s~U(0.25,1)`，无 blur；日志 `deg_scale_mean≈0.624/0.625` 正常。consistency 是 clean logits detach 作为 teacher，degraded rec logits 做 KL，`tau=4`，乘 `tau^2`，形式上是标准 KD/consistency loss。iso 下 `rec_only=True` 会在 stage-3 fork 处 detach，f_full BNNeck 不跑 degraded 图，未看到明显梯度泄漏。
./reassess/r_3.md:7957:./reassess/r_3.md:4100:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:155:代码主路径基本可信，尤其 `iso_trunk_recce=0 / rec_only=True` 的梯度隔离没有看到实质性泄漏。但结果解释需要收紧：AG-ReID.v2 final 日志不支持“net gain ≈ +1.5 / 等于融合增益”，部分文档还把 ep50 当成 final，且 seed2 / 3090 AIRL 日志不完整。
./reassess/r_3.md:7958:./reassess/r_3.md:4101:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/_verdicts.txt:169:  `full_map.detach()` 的位置足以切断梯度，但它发生在 f_full Swin late stages 和 `layer4` 已经执行之后。也就是说 degraded `rec_only` pass 仍计算了 f_full 激活并消耗随机层 RNG；只是这些结果不进 loss、不更新 f_full BNNeck/权重。这是注释“cheap / no f_full grad graph built”的夸大，不是梯度泄漏。
./reassess/r_3.md:7961:./reassess/r_3.md:4138:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_1.md:7276:../new_angle_AIRL.md:169:2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一致(同一 degrade+forward 原语), 刻意保持对齐以保证消融诚实; 是否有实质影响由 kill-switch #4 训练结果裁决, 非 bug。若要彻底隔离需加 rec-only forward(改训练行为, 当前 lab-3090 正在跑此代码, 不动)。
./reassess/r_3.md:7988:./reassess/r_3.md:4328:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_5.md:9912:../new_angle_AIRL.md:169:2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一致(同一 degrade+forward 原语), 刻意保持对齐以保证消融诚实; 是否有实质影响由 kill-switch #4 训练结果裁决, 非 bug。若要彻底隔离需加 rec-only forward(改训练行为, 当前 lab-3090 正在跑此代码, 不动)。
./reassess/r_3.md:7989:./reassess/r_3.md:4370:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:51:3. **compositional 组合泛化 + group-DRO（exp330，搬 Sagawa）→ NO-GO**：(遮挡物类×身体部位) held-out 组合，赌模型学 occluder 捷径会崩。**ERM held-out vs seen GAP=+0.10≈0 → 模型本就组合鲁棒**（不学 occluder 捷径）。双审+smoke 全过、单变量干净，kill-switch 便宜判死。
./reassess/r_3.md:7993:./reassess/r_3.md:4481:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:14:"transfer_to_occluded": "Adapt the VCS \"corrupt-a-parsed-region + feature-consistency loss\" trick to occlusion: use human parsing (or an occluder mask) to identify non-target/occluded regions, synthesize occlusion by replacing those pixels with random pixels sampled from a per-batch pool, then add an L2/MSE alignment loss pulling the corrupted-image feature toward the clean-image feature. This is a training-only, self-supervised occlusion-augmentation regularizer that teaches the backbone to produce features invariant to corrupted/missing regions, without needing extra inference modules or pose at test time. The batch-pool pixel-shuffle (rather than zero/black masking) is the transferable detail — it injects realistic distractor texture instead of constant fills, which better mimics real occluders.",
./reassess/r_3.md:7994:./reassess/r_3.md:4482:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:21:"mechanism": "TTPM is a ViT + HRNet pipeline with three parts and no occlusion-sample augmentation. A Multi-patch Feature Encoder encodes intra-patch and inter-patch features (re-grouping patches at 2x scale with the cls token). Pose-Patch Mapping (PPM) builds a fine-grained, per-patch positional correspondence between M pose keypoints and N patches via a learned similarity S = gamma*D_M - eta*S_M that combines an affine-extended Mahalanobis distance and affine-extended cosine similarity, gates out low-confidence keypoints by a heatmap-confidence threshold, and selects the argmax-similarity patch per keypoint, thereby down-weighting occluded patches without coarse global pose mapping. A Texture-Aware Decoder (context-extraction MHSA layers + texture-extraction MHCA layers using pose-filtered features as queries) leverages texture distinctiveness to separate the target from non-target pedestrian occluders. An auxiliary pose loss penalizes high cosine similarity between human-part and non-human (background) features.",
./reassess/r_3.md:7995:./reassess/r_3.md:4483:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:22:"transfer_to_occluded": "Two directly usable ideas for occluded ReID. (1) The pose loss L_p that explicitly minimizes cosine similarity between body-part features and non-human/background features — a cheap regularizer to suppress occluder/background leakage into the identity embedding, applicable to any pose- or part-based occluded model. (2) The texture-distinctiveness decoder channel aimed specifically at non-target PEDESTRIAN occlusion: when the occluder is itself a person, pose/structure cues fail (same skeleton topology), so using texture appearance distinctiveness (via pose-filtered queries cross-attending to contextual features) to push apart target vs distractor-person is a concrete mechanism for the under-addressed person-on-person occlusion case. Also transferable: replacing coarse pose-to-global mapping with a per-keypoint-to-patch confidence-gated argmax correspondence (Mahalanobis+cosine affine similarity) to localize visible regions.",
./reassess/r_3.md:7999:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_3.md:8003:./reassess/r_3.md:4491:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:86:"transfer_to_occluded": "Apply the text-level orthogonal-projection disentanglement to occlusion: learn a dual prompt for the OCCLUDER/distractor (e.g. \"a photo of the [X]...[X] occluder\") alongside the ID prompt, then orthogonalize the occluder text feature out of the ID text feature (F_ort = F_id - proj of F_occ onto F_id) and use the purged ID text feature to guide the visual encoder via the non-local SIM. Because the non-local operation aggregates from spatially distant but semantically relevant regions, it is well suited to occluded ReID where visible body parts are scattered far from the occluded/corrupted region — letting the model pull ID evidence from non-adjacent visible patches rather than local self-attention that gets dominated by the occluder. The orthogonalization gives a clean way to subtract a known nuisance direction (occluder appearance) without an auxiliary parser.",
./reassess/r_3.md:8004:./reassess/r_3.md:4492:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:88:"novel_component": "Disentangling a nuisance factor by vector orthogonal projection in the TEXT embedding space: project the nuisance (clothing) text feature onto the ID text direction and subtract it, purging the nuisance-aligned component of ID features without any auxiliary visual model — a cheap, parser-free disentanglement that generalizes to any named confounder (including occluders)."
./reassess/r_3.md:8005:./reassess/r_3.md:4493:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:102:"transfer_to_occluded": "Use real out-of-domain vehicle (or generic object) crops as the occluder source for copy-paste augmentation instead of synthetic gray patches or random erasing, and inject them with a controlled height-ratio so the paste lands on the lower body to mimic real vehicle-induced occlusion. This produces texture-and-color-realistic distractor occluders that better match deployment-time occlusions than random-erasing, and the epoch-wise alternation between clean and occluder-domain data acts as a regularizer pushing the model toward occlusion-invariant features — both are drop-in, annotation-free, and backbone-agnostic for any occluded ReID pipeline.",
./reassess/r_3.md:8010:./reassess/r_3.md:4500:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:142:"transfer_to_occluded": "Use an off-the-shelf 3D body reconstruction (SMPL/HMR) of the pedestrian to generate geometric reference points, then drive deformable cross-attention so the image-feature sampling locations are anchored at projected body-keypoint/mesh positions. For occluded ReID this gives a body-structure prior independent of pixel evidence: even when a region is occluded, the reconstructed mesh still provides a body-anchored query, steering attention onto plausible person regions and away from distractor occluders, and the pose/shape-derived reference points act as a soft visibility/structure guide. The SSTI idea of an appearance-independent shape descriptor (from β only) is also a complementary, occlusion-robust feature since estimated body shape is less corrupted by occluders than RGB texture.",
./reassess/r_3.md:8021:./reassess/r_3.md:4649:/Users/abdslm/Desktop/SOLIDER-REID/experiments/publishable_angles_synthesis.md:40:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:8030:./reassess/r_3.md:4719:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:133:- **直接坐实 oracle agent 的判断**：「frozen-source 换源(RADIO/SAM2/DIFT/DINOv3)价值大降」现在有**全量实证**——换更新 DINO（registers）只 +0.29，换 SD 范式（DIFT）反而 −1.13。**换源都不破 frozen 天花板，瓶颈在 frozen 本身**。DINO 线唯一活口仍是 exp324d LoRA（解冻），与 oracle 结论收敛。
./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
./reassess/r_3.md:8057:./pivot/p_2.md:700:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:8061:./pivot/p_2.md:835:./analyses/deep_4.txt:53:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8062:./pivot/p_2.md:836:./analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8069:./pivot/p_2.md:1061:./analyses/ALL_ANALYSES.txt:1524:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8070:./pivot/p_2.md:1062:./analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8072:./pivot/p_2.md:1493:./reviews/deep_4.md:11383:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8073:./pivot/p_2.md:1494:./reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8074:./pivot/p_2.md:1500:./reviews/deep_4.md:11460:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8075:./pivot/p_2.md:1501:./reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8092:./pivot/p_2.md:2277:./validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:8093:./pivot/p_2.md:2278:./validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:8097:./pivot/p_2.md:3325:./validate/v_3.md:15168:../litreview2/validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:8098:./pivot/p_2.md:3647:./validate/v_3.md:15494:../litreview2/validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:8099:./pivot/p_2.md:3955:./validate/v_3.md:16098:../litreview2/validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:8112:./pivot/p_2.md:5136:pivot/p_3.md:1994:### ⚠️死亡清单碰撞(排除): 07的CIL-ReID=已kill 3次的donor-leakage; AGP/RWOA=遮挡红海pose/CLIP/part二次包装。
./reassess/r_3.md:8117:./pivot/p_2.md:5143:pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
./reassess/r_3.md:8125:./pivot/p_2.md:5320:reviews/deep_11.md:740:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8126:./pivot/p_2.md:5378:reviews/deep_24.md:403:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8128:./pivot/p_2.md:5422:reviews/deep_24.md:976:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:8131:./pivot/p_2.md:5570:reviews/deep_5.md:373:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8135:./pivot/p_2.md:5735:pivot/p_2.md:835:./analyses/deep_4.txt:53:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8136:./pivot/p_2.md:5736:pivot/p_2.md:836:./analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8140:./pivot/p_2.md:5782:pivot/p_2.md:1061:./analyses/ALL_ANALYSES.txt:1524:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8141:./pivot/p_2.md:5783:pivot/p_2.md:1062:./analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8142:./pivot/p_2.md:5858:pivot/p_2.md:1493:./reviews/deep_4.md:11383:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8143:./pivot/p_2.md:5859:pivot/p_2.md:1494:./reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8144:./pivot/p_2.md:5863:pivot/p_2.md:1500:./reviews/deep_4.md:11460:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8145:./pivot/p_2.md:5864:pivot/p_2.md:1501:./reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8146:./pivot/p_2.md:5971:pivot/p_2.md:2277:./validate/v_2.md:7033:validate/v_3.md:4047:../litreview/reviews/lit_15.md:4238:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:8147:./pivot/p_2.md:5972:pivot/p_2.md:2278:./validate/v_2.md:7035:validate/v_3.md:4063:../litreview/reviews/lit_15.md:4294:切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
./reassess/r_3.md:8161:./pivot/p_2.md:6481:reviews/deep_18.md:745:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8168:./pivot/p_2.md:6632:reviews/deep_4.md:11383:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8169:./pivot/p_2.md:6633:reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8170:./pivot/p_2.md:6643:reviews/deep_4.md:11460:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8171:./pivot/p_2.md:6644:reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8174:./pivot/p_2.md:6747:reviews/deep_28.md:869:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8179:./pivot/p_2.md:6913:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:8182:./pivot/p_2.md:6924:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:8199:./pivot/p_2.md:7139:reviews/deep_6.md:699:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:8200:./pivot/p_2.md:7142:reviews/deep_6.md:757:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8201:./pivot/p_2.md:7166:reviews/deep_20.md:387:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8203:./pivot/p_2.md:7210:reviews/deep_20.md:960:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:8217:./pivot/p_2.md:7336:reviews/deep_23.md:3311:前两篇已经读到核心：一篇把遮挡解释成真实遮挡分布、文本语义缺失和增强后特征分布不一致三件事；另一篇把退化 ReID 从“单一退化域偏移”改写成“混合退化下的分布偏移加身份拓扑扭曲”。我会补一下被输出截断的中间方法细节，然后继续后四篇。
./reassess/r_3.md:8219:./pivot/p_2.md:7352:reviews/deep_23.md:8955:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:8221:./pivot/p_2.md:7370:reviews/deep_23.md:9005:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:8223:./pivot/p_2.md:7392:reviews/deep_2.md:432:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8224:./pivot/p_2.md:7433:reviews/deep_2.md:874:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:8229:./pivot/p_2.md:7502:reviews/deep_17.md:398:- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。
./reassess/r_3.md:8232:./pivot/p_2.md:7532:reviews/deep_9.md:512:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8236:./pivot/p_2.md:7623:reviews/deep_27.md:648:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8248:./pivot/p_2.md:7800:reviews/deep_16.md:588:审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。
./reassess/r_3.md:8261:./pivot/p_2.md:7923:analyses/deep_23.txt:6:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:8262:./pivot/p_2.md:7941:analyses/deep_4.txt:53:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8263:./pivot/p_2.md:7942:analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8266:./pivot/p_2.md:7972:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:8286:./pivot/p_2.md:8129:analyses/ALL_ANALYSES.txt:1016:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:8288:./pivot/p_2.md:8154:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:8291:./pivot/p_2.md:8177:analyses/ALL_ANALYSES.txt:1524:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8292:./pivot/p_2.md:8178:analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8298:./pivot/p_2.md:8427:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:8349:/Users/abdslm/Desktop/SOLIDER-REID/experiments/results.md:336:**结论**: PSG 在所有数据集和 backbone 上均有效。在遮挡数据集上增益更大。
./reassess/r_3.md:8361:/Users/abdslm/Desktop/SOLIDER-REID/experiments/results.md:1667:- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。
./reassess/r_3.md:8363:/Users/abdslm/Desktop/SOLIDER-REID/experiments/results.md:1755:| exp354 PC-SOR | pose+CLIP文本 token归属(20-codex首推) | kill-switch FAILED | CLIP文本定位不了遮挡物/分不清目标vs任意人 |
./reassess/r_3.md:8373:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:1491:- 🔴 红队（方案 A）核心论点: SGMKC 实现成本极低（~15 行代码，无新参数），属于不同类别的改进（训练方法论 vs 架构/loss 添加），skeleton graph + masked prediction 组合是真正新颖的（FCFormer 用 transformer decoder，MAE 用 random patches，没有人在 skeleton graph 上做过 masked completion for ReID）。47 个失败实验都是架构添加或 loss 变体，SGMKC 是 self-supervised 训练策略——不同搜索空间。即使失败，负面结论也有论文价值。信心: 6/10
./reassess/r_3.md:8383:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:2016:4. 因而 `exp109` 的高价值结论仍应保留：真正缺的不是再换一个 loss 形式，而是 **更 support-complete 的 teacher**。
./reassess/r_3.md:8386:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:2054:3. 结合 `exp109` 的 oracle 结论，更合理的新解释是：
./reassess/r_3.md:8405:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:2919:- `exp109` 的 oracle 结论始终没有被推翻：
./reassess/r_3.md:8410:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:2952:3. 这与 SGMKC, SCRC, SCKD 的结论一致：feature-level completion 在 15K 数据集上不可行
./reassess/r_3.md:8434:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:4550:**上下文**: 验证 VC-Norm(occlusion-as-domain-factor, visibility-conditioned normalization) 跨域创新的**前提**——遮挡是否在 per-part-token 的 normalization statistic(mean/var) 上造成可分离分布 shift。若 KL≈0 则无燃料 kill。lab-3090-d 无训练 probe。
./reassess/r_3.md:8444:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:4588:**下一步**: cheap/vetted 空间穷尽。VC-Norm 跨域(慢, ~8h)是唯一活线。剩重量级 import(mmcv-gated/低EV)留用户定。真实交付=诊断论文(8 kill + 吸收陷阱 + 张力 + 三堵墙)。
./reassess/r_3.md:8446:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:4593:**决策**: KILL VC-Norm。9 个 bet 全 NO-GO。**真实交付=诊断/analysis 论文**(9 kill + 吸收陷阱 + 张力 + 三堵墙)。
./reassess/r_3.md:8447:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:4603:**决策**: **纯 CLIP 文本部位路线=死路**(语义冗余)。step2 的新 CLIP 接法必须带 global 没有的真信息(CLIP 视觉特征/遮挡推理/ID 级原型)。理想判据(用户定):baseline 58 → +CLIP 59(CLIP 单独过坎)→ +pose-CLIP 60;现状是 CLIP 单独那步=0。
./reassess/r_3.md:8450:本地 GOPL kill-switch 设计已经明确把“2D 可见性解释同样强”列为致命红线；用户给出的新结果正好击中这条红线。遮挡 source-separation 只有一条很窄的生路：它必须把隐藏变量从“目标缺失多少”换成“遮挡者身份向量具体污染了谁”，并且压掉这部分后直接提高 ReID，而不是又回到 occlusion-count。
./reassess/r_3.md:8467:一个做行人 ReID 的团队要发 CCF-B **方法稿**, 刚连续用零训练 kill-switch 干净杀掉两个方向:
./reassess/r_3.md:8476:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_3.md:8519:-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
./reassess/r_3.md:8680:2. **重定义动作**：作者把问题讲成“跨模态对齐缺少显式、模态无关的 semantic anchor”，而不是“再做一个 CLIP prompt”。关键词是 attribute-aware text、contour-aware text prompt、text re-definition、category-level textual prototypes、modality-agnostic semantic anchors。这个重定义把文本从辅助描述变成“RGB 和 IR 共同对齐的类别原型”。
./reassess/r_3.md:8682:3. **机制怎么长出来**：ATG 用 BLIP 问答和固定模板生成属性文本，并刻意避开颜色，保留 age、gender、action、clothing type、hair、glasses 这些更跨模态的属性。CTP 用 SCHP 人体解析去掉衣服区域，保留 body shape 和 pose 相关区域，再用可学习 prompt 表示 contour-aware class text。TRM 负责把 instance-level attribute text 和 class-level contour prompt 融成 image-aware redefined text，最后做 class-level text prototype，指导 RGB 和 IR visual feature 对齐。机制和“属性语义加结构语义”的问题定义是匹配的，但整体偏组合式。
./reassess/r_3.md:8684:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./reassess/r_3.md:8686:5. **reviewer 为什么买账**：这篇卖的主要是视角，不是模块精巧度。reviewer 买账点是它把“文本提示”拆成两类可解释语义：属性和轮廓，并用 prototype 把文本变成跨模态锚点。novelty 的来源是“显式语义原型替代弱 token proxy”，而不是 BLIP、SCHP、CLIP 或 Transformer decoder 本身。
./reassess/r_3.md:8690:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./reassess/r_3.md:8692:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./reassess/r_3.md:8694:3. **机制怎么长出来**：SRDA 很直接：先用 MLLM 判断性别，在同一性别内随机选两个 ID，把一个人的上半身和另一个人的下半身横向拼接，生成新 ID，再用微调 BLIP 为新图生成文本。它不追求自然，而追求结构组合多样性。CSRM 则在测试时为文本 query 找 cross-modal k-reciprocal 图像邻居，用 Jaccard 权重把可靠图像特征融合进文本特征，再检索。训练阶段用 CLIP 双流、ID loss、SDM loss 和 image triplet；不加 text triplet，因为 MLLM 文本有噪声。
./reassess/r_3.md:8696:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./reassess/r_3.md:8704:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:8706:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:8710:5. **reviewer 为什么买账**：这篇卖的是“对遮挡失败类型的细分”加“机制一一对应”。如果只说 pose+texture，novelty 会弱；但作者先指出 pose 方法在极端遮挡和行人遮挡下分别为什么失效，再让 PPM 和 TAD 分别处理，reviewer 就更容易接受。真正 novelty 来自问题拆解，而不是 Transformer、pose 或 attention 本身。
./reassess/r_3.md:8735:analyses/deep_15.txt:32:1. **触发观察**：作者先展示了文本行人检索的典型失败：错误结果排在前面，不是因为整体更像，而是颜色对上了。比如文本写“blue t-shirt”，目标其实是很浅的蓝，模型却召回一堆明显蓝色衣服的人。RGB 和灰度模型的响应图也显示它们关注不同区域。
./reassess/r_3.md:8738:analyses/deep_15.txt:40:2. **重定义动作**：它把问题从“合成数据规模不够大”改成“合成数据的泛化能力由纹理真实性和监控采集协议决定”。关键词是 `diffusion-generated UV texture`、`reliable clothing texture`、`sub-monitoring network`、`scalable synthetic pipeline`、`direct transfer`。大家以为扩大身份数和场景数就能提升合成数据，其实 UV 纹理定义和监控网络模拟才是关键瓶颈。
./reassess/r_3.md:8739:analyses/deep_15.txt:41:3. **机制怎么长出来**：MakeHuman 生成不同体型、肤色、发型的人体模型，再加骨架和行走动画。衣服模型本身没有足够纹理，于是用 normal maps 条件训练 ControlNet，引导 Stable Diffusion 生成和衣服表面结构对齐的 UV texture maps，再导入 UE4 材质系统。采集上设计 sub-monitoring network，每个子网络用少量摄像头覆盖局部区域，并让人物沿循环路径移动，控制视角、背景、光照和遮挡。UnrealCV 自动得到实例分割和裁剪。
./reassess/r_3.md:8749:analyses/deep_23.txt:4:1. **触发观察**：作者抓住三个具体失败点。遮挡会造成视觉语义缺失，CLIP-ReID 的图文全局对齐在遮挡下会错位，常用随机擦除默认遮挡均匀分布，但真实遮挡更多集中在下半身，尤其腿部。
./reassess/r_3.md:8751:analyses/deep_23.txt:6:3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
./reassess/r_3.md:8752:analyses/deep_23.txt:7:4. **证据闭环**：关键不是只报总成绩，而是三类消融分别对应三类重定义。`Baseline+RPE` 在 Occluded-Duke 上大幅超过普通随机擦除，证明“遮挡分布先验”有用；`Baseline+DMPA` 和去掉 DMPA 的版本证明文本语义稳态不是装饰；MDO 的 only enhancement 和完整 MDO 对比证明两步对抗比普通扰动更有效。
./reassess/r_3.md:8756:analyses/deep_23.txt:19:2. **重定义动作**：它把换衣问题从“消除衣服”改写成“分离负语义并显式挖出正语义”。关键词是 `negative factors`、`positive factors`、`semantic contextual integration`、`semantic separation`、`orthogonalized text features`。
./reassess/r_3.md:8757:analyses/deep_23.txt:20:3. **机制怎么长出来**：SSE 用两个 prompt，一个描述 person，一个描述 clothes。先得到身份文本特征和衣服文本特征，再把衣服方向从身份方向里投影并减掉，得到去衣服后的语义特征。SIM 再让这个正交化文本特征通过非局部操作和跨注意力去指导视觉特征。这个设计和“负语义分离、正语义引导”的重定义绑定很紧。
./reassess/r_3.md:8758:analyses/deep_23.txt:21:4. **证据闭环**：它的关键消融是 SSE 单独有效，SIM 单独在 CNN 上甚至会伤性能，SSE 加 SIM 才最好。这证明“文本指导本身不够，必须先把衣服语义分离干净”。t-SNE 更紧凑、prompt 引导热图更关注头肩、下身、鞋，特征相似度分布在正交化后更清晰，这些可视化都在支撑“语义分离是对的”。
./reassess/r_3.md:8759:analyses/deep_23.txt:22:5. **审稿人为什么买账**：这篇卖的是视角。CLIP-ReID 已经有 ID prompt，CAL 已经有衣服对抗，但它把换衣讲成“负因素和正因素同时存在”的语义问题，用文本空间正交化给出可解释动作。新意不是“用了 CLIP”，而是把衣服偏置从语义空间里切出来，再反过来指导视觉。
./reassess/r_3.md:8767:analyses/deep_22.txt:9:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./reassess/r_3.md:8769:analyses/deep_4.txt:53:1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。
./reassess/r_3.md:8770:analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:8771:analyses/deep_4.txt:57:3. 机制怎么长出来：机制基本是顺着这个重定义长出来的。主体 token S* 来自全局图像特征，表示“这是谁”；属性 token A* 由 learnable queries 从 patch 中提取可见身份属性；上下文 token O* 提取遮挡物特征。三者组成 “A photo of a [S*] person with [A*] partially occluded by [O*]” 这样的细粒度 prompt，再用文本作为 query 对视觉 patch 做 cross-attention，引导图像特征看可见人体区域、压掉遮挡物。动态融合用预测不确定性给 visual、text、cross-attention feature 分配权重，是对可靠性问题的补强。
./reassess/r_3.md:8782:analyses/deep_3.txt:31:3. 机制怎么长出来：第一阶段冻结 CLIP 图文编码器，用 CoOp 学属性 prompt 的可学习 token，把这些 token 当属性伪标签，同时训练 attribute encoder 从图像特征预测这些 token。第二阶段用整体文本特征和属性伪标签双重引导图像编码器，再用 noise-mask 跳过遮挡导致的不可靠属性匹配。
./reassess/r_3.md:8798:analyses/deep_25.txt:45:3. **机制怎么长出来**：ATG 用 BLIP 问答和固定模板生成属性文本，并刻意避开颜色，保留 age、gender、action、clothing type、hair、glasses 这些更跨模态的属性。CTP 用 SCHP 人体解析去掉衣服区域，保留 body shape 和 pose 相关区域，再用可学习 prompt 表示 contour-aware class text。TRM 负责把 instance-level attribute text 和 class-level contour prompt 融成 image-aware redefined text，最后做 class-level text prototype，指导 RGB 和 IR visual feature 对齐。机制和“属性语义加结构语义”的问题定义是匹配的，但整体偏组合式。
./reassess/r_3.md:8800:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:8802:analyses/deep_25.txt:73:5. **reviewer 为什么买账**：这篇卖的是“对遮挡失败类型的细分”加“机制一一对应”。如果只说 pose+texture，novelty 会弱；但作者先指出 pose 方法在极端遮挡和行人遮挡下分别为什么失效，再让 PPM 和 TAD 分别处理，reviewer 就更容易接受。真正 novelty 来自问题拆解，而不是 Transformer、pose 或 attention 本身。
./reassess/r_3.md:8862:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:4588:**下一步**: cheap/vetted 空间穷尽。VC-Norm 跨域(慢, ~8h)是唯一活线。剩重量级 import(mmcv-gated/低EV)留用户定。真实交付=诊断论文(8 kill + 吸收陷阱 + 张力 + 三堵墙)。
./reassess/r_3.md:8864:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:4593:**决策**: KILL VC-Norm。9 个 bet 全 NO-GO。**真实交付=诊断/analysis 论文**(9 kill + 吸收陷阱 + 张力 + 三堵墙)。
./reassess/r_3.md:8865:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:4603:**决策**: **纯 CLIP 文本部位路线=死路**(语义冗余)。step2 的新 CLIP 接法必须带 global 没有的真信息(CLIP 视觉特征/遮挡推理/ID 级原型)。理想判据(用户定):baseline 58 → +CLIP 59(CLIP 单独过坎)→ +pose-CLIP 60;现状是 CLIP 单独那步=0。
./reassess/r_3.md:8868:本地 GOPL kill-switch 设计已经明确把“2D 可见性解释同样强”列为致命红线；用户给出的新结果正好击中这条红线。遮挡 source-separation 只有一条很窄的生路：它必须把隐藏变量从“目标缺失多少”换成“遮挡者身份向量具体污染了谁”，并且压掉这部分后直接提高 ReID，而不是又回到 occlusion-count。
./reassess/r_3.md:8885:一个做行人 ReID 的团队要发 CCF-B **方法稿**, 刚连续用零训练 kill-switch 干净杀掉两个方向:
./reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
./reassess/r_3.md:8937:-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
./reassess/r_3.md:9133:pivot/p_2.md:836:./analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9139:pivot/p_2.md:1062:./analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9142:pivot/p_2.md:1494:./reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9143:pivot/p_2.md:1501:./reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9150:pivot/p_2.md:1861:./reviews/deep_3.md:5457:   419	matches plus activation maps from ResNet-50 model. Bottom: Our 3D approach uses UVTexture mapping with texture, visibility masks, and activation
./reassess/r_3.md:9165:pivot/p_2.md:3325:./validate/v_3.md:15168:../litreview2/validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9166:pivot/p_2.md:3647:./validate/v_3.md:15494:../litreview2/validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9167:pivot/p_2.md:3955:./validate/v_3.md:16098:../litreview2/validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9183:pivot/p_2.md:5736:pivot/p_2.md:836:./analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9185:pivot/p_2.md:5783:pivot/p_2.md:1062:./analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9186:pivot/p_2.md:5859:pivot/p_2.md:1494:./reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9187:pivot/p_2.md:5864:pivot/p_2.md:1501:./reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9194:pivot/p_2.md:6633:reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9195:pivot/p_2.md:6644:reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9197:pivot/p_2.md:6913:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9200:pivot/p_2.md:6924:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9216:pivot/p_2.md:7942:analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9219:pivot/p_2.md:7972:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9228:pivot/p_2.md:8154:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9231:pivot/p_2.md:8178:analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9249:validate/v_3.md:1887:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9250:validate/v_3.md:1888:./reviews/deep_25.md:14099:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:9253:validate/v_3.md:1906:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9254:validate/v_3.md:1907:./reviews/deep_25.md:14178:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:9256:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9261:validate/v_3.md:2945:../litreview2/analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9262:validate/v_3.md:2946:../litreview2/analyses/ALL_ANALYSES.txt:1186:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:9264:validate/v_3.md:3005:../litreview2/analyses/ALL_ANALYSES.txt:1462:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9266:validate/v_3.md:3019:../litreview2/analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9307:validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9308:validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9312:validate/v_3.md:14184:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9313:validate/v_3.md:15168:../litreview2/validate/v_3.md:7368:./validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9314:validate/v_3.md:15494:../litreview2/validate/v_3.md:8050:./validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9315:validate/v_3.md:16098:../litreview2/validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9326:validate/v_2.md:1307:./reviews/deep_23.md:664:misalignment between the visual and textual features caused by occlusions. These challenges motivate us
./reassess/r_3.md:9367:validate/v_2.md:2455:./reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:9388:validate/v_2.md:5109:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9389:validate/v_2.md:5121:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9391:validate/v_2.md:5173:./analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9395:validate/v_2.md:5932:./analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9407:validate/v_2.md:6542:./reviews/deep_1.md:4495:/bin/zsh -lc 'pdftotext "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n -C 2 "Ablation|ablation|visual|Visualization|t-SNE|CMF|SSTI|3D|shape|occlusion|misalignment"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9441:validate/v_2.md:7883:validate/v_2.md:2455:./reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:9467:validate/v_2.md:9360:reviews/deep_21.md:6604:   126	color and texture, while in infrared images, occlusions often
./reassess/r_3.md:9471:validate/v_2.md:10892:reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:9480:validate/v_2.md:12726:validate/v_3.md:2754:../codex_review_ovli.txt:3959:experiments/overnight_innovation_log.md:51:3. **Frozen partial/unbalanced-OT**：把 part-MaxSim 的平均换成 5×5 Sinkhorn（可见性当 marginal，occlusion=mass destruction）。angle-4 kill-switch。
./reassess/r_3.md:9506:reviews/deep_3.md:2932:    57	However, in the context of occlusion ReID tasks, these models face
./reassess/r_3.md:9517:reviews/deep_3.md:5063:    25	approaches that simply overlay textures on point clouds, our method exploits the full potential of UVTexture
./reassess/r_3.md:9535:reviews/deep_3.md:5268:   230	For creating 3D textures via UVTexture maps for humans, we adopt
./reassess/r_3.md:9564:reviews/deep_3.md:5457:   419	matches plus activation maps from ResNet-50 model. Bottom: Our 3D approach uses UVTexture mapping with texture, visibility masks, and activation
./reassess/r_3.md:9608:reviews/deep_3.md:6764:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9611:reviews/deep_3.md:6819:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9614:analyses/deep_28.txt:77:它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。
./reassess/r_3.md:9620:reviews/deep_15.md:3020:3) UV Texture Maps: UV texture is a technique that ensures
./reassess/r_3.md:9626:reviews/deep_15.md:5439:   312	3) UV Texture Maps: UV texture is a technique that ensures
./reassess/r_3.md:9635:reviews/deep_15.md:12108:   312	3) UV Texture Maps: UV texture is a technique that ensures
./reassess/r_3.md:9666:analyses/deep_3.txt:46:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9674:reviews/deep_21.md:306:/bin/zsh -lc "pdftotext -f 1 -l 8 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | rg -n \""'^('"\\s*[0-9IVX]+\\.?\\s+)?(METHOD|Method|METHODOLOGY|Methodology|Approach|APPROACH|Proposed|Framework|Experiments|Experiment|RELATED|Related|III\\.|3\\.|4\\.)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9698:reviews/deep_21.md:1788:/bin/zsh -lc "pdftotext -f 1 -l 8 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '250,720p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9715:reviews/deep_21.md:3007:/bin/zsh -lc "pdftotext -f 1 -l 8 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '716,900p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9736:reviews/deep_21.md:4003:/bin/zsh -lc "pdftotext -f 9 -l 12 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '1,320p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9750:reviews/deep_21.md:5169:/bin/zsh -lc "pdftotext -f 12 -l 15 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9790:reviews/deep_21.md:6604:   126	color and texture, while in infrared images, occlusions often
./reassess/r_3.md:9793:reviews/deep_21.md:6634:   156	occlusion from a completion perspective rather than a discarding one. Specifically, we propose to fully exploit both intramodality contextual cues and inter-modality complementary
./reassess/r_3.md:9799:reviews/deep_21.md:6704:   226	information with visual features to enhance contextual representation under occlusion. Chen et al. [17] introduced a
./reassess/r_3.md:9804:reviews/deep_21.md:7393:/bin/zsh -lc "pdftotext -f 1 -l 4 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9843:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:9844:analyses/deep_25.txt:69:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:9848:analyses/deep_4.txt:55:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:9858:reviews/deep_25.md:6706:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9867:reviews/deep_25.md:6745:pedestrian occlusion, a Texture-Aware Decoder is introduced to perceive texture features and leverage their
./reassess/r_3.md:9901:reviews/deep_25.md:6852:non-targets. As the appearance of target and non-target pedestrians often exhibits notable texture variations, it is more feasible to leverage
./reassess/r_3.md:9916:reviews/deep_25.md:6961:pose-patch mapping and introduce a texture-aware decoder to realize
./reassess/r_3.md:9922:reviews/deep_25.md:7013:through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.
./reassess/r_3.md:9953:reviews/deep_25.md:7901:Contribution Analysis of TAD Components on OccludedDuke (%). TELs and CELs denote Texture extraction layers and Context extraction layers in TAD. Bold indicates
./reassess/r_3.md:9966:reviews/deep_25.md:8155:texture-aware encoder (left column), TTPM accurately re-identiﬁes the
./reassess/r_3.md:9969:reviews/deep_25.md:8160:textures. This result indicates that the proposed TTPM still has limitations in relying on texture features. While these features enable accurate
./reassess/r_3.md:9973:reviews/deep_25.md:11002:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Overview|Multi|Pose|Texture|Loss|Ablation|Effectiveness|Comparison)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9983:reviews/deep_25.md:11862:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | sed -n '293,773p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:9989:reviews/deep_25.md:11877:through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.
./reassess/r_3.md:10000:reviews/deep_25.md:12347:TTPM 的页面排版导致方法文本顺序有些乱，刚才截取漏了 MFE 的前半段。我会再补一次该论文从 3.1 到 3.3 的开头，避免把 pose-patch mapping 的逻辑链断掉。
./reassess/r_3.md:10001:reviews/deep_25.md:12353:/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | sed -n '174,400p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10008:reviews/deep_25.md:12435:pose-patch mapping and introduce a texture-aware decoder to realize
./reassess/r_3.md:10014:reviews/deep_25.md:12487:through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.
./reassess/r_3.md:10023:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:10024:reviews/deep_25.md:14099:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:10028:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:10029:reviews/deep_25.md:14178:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:10044:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:10045:analyses/ALL_ANALYSES.txt:1186:3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。
./reassess/r_3.md:10047:analyses/ALL_ANALYSES.txt:1403:它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。
./reassess/r_3.md:10048:analyses/ALL_ANALYSES.txt:1462:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:10051:analyses/ALL_ANALYSES.txt:1526:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:10077:reviews/deep_28.md:6930:它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。
./reassess/r_3.md:10078:reviews/deep_28.md:7020:它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。
./reassess/r_3.md:10208:reviews/deep_4.md:788:with a occlusion-aware fine-grained textual prompt in the format:”A photo of a [S ∗ ] person with [A∗ ] partially occluded
./reassess/r_3.md:10233:reviews/deep_4.md:11385:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:10234:reviews/deep_4.md:11462:2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。
./reassess/r_3.md:10274:reviews/deep_20.md:2165:    17	and occlusions, by capturing both fine-grained details and broad contextual cues. Additionally,
./reassess/r_3.md:10324:reviews/deep_20.md:5980:   748	In the context of occluded person re-identification, occlusions increase task entropy because visual cues become incomplete and ambiguous. By injecting structured ‘‘vehicle noise’’ in the form of semantically coherent patterns (e.g., vehicle textures and contours), the
./reassess/r_3.md:10391:reviews/deep_1.md:684:/bin/zsh -lc 'pdftotext -f 1 -l 8 "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10392:reviews/deep_1.md:1055:/bin/zsh -lc 'pdftotext -f 1 -l 8 "A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10395:reviews/deep_1.md:2096:/bin/zsh -lc "pdftotext -f 1 -l 8 \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '1,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10418:reviews/deep_1.md:4495:/bin/zsh -lc 'pdftotext "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n -C 2 "Ablation|ablation|visual|Visualization|t-SNE|CMF|SSTI|3D|shape|occlusion|misalignment"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10434:reviews/deep_1.md:5191:/bin/zsh -lc "pdftotext \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '880,1058p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10435:reviews/deep_1.md:5378:/bin/zsh -lc "pdftotext -layout \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '900,1005p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10436:reviews/deep_1.md:5492:/bin/zsh -lc "pdftotext -layout -f 7 -l 9 \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '1,240p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10443:reviews/deep_1.md:7603:/bin/zsh -lc "pdftotext -f 1 -l 8 \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | sed -n '1,675p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10471:reviews/deep_1.md:7740:generates semantically relevant occlusion patches with semantic information of object pedestrians and further realizes occlusion simulation based on their position, shape, and context to generate more
./reassess/r_3.md:10528:reviews/deep_1.md:8292:/bin/zsh -lc "pdftotext -layout -f 7 -l 11 \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | sed -n '1,500p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10548:reviews/deep_1.md:8665:/bin/zsh -lc "pdftotext \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | rg -n -C 3 \"Ablation|ablation|SGOS|LFSE|cross|offset|occlusion|semantic|visual|Fig\\. 5|heatmap|t-SNE|Grad|effectiveness|component\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10576:reviews/deep_1.md:8776:136:generates semantically relevant occlusion patches with semantic information of object pedestrians and further realizes occlusion simulation based on their position, shape, and context to generate more
./reassess/r_3.md:10736:reviews/deep_14.md:1287:161:with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reassess/r_3.md:10782:reviews/deep_14.md:3974:   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reassess/r_3.md:10792:reviews/deep_14.md:5763:161:   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
./reassess/r_3.md:10804:reviews/deep_16.md:1194:/bin/zsh -lc "pdftotext -f 1 -l 8 'Instant pose extraction based on mask transformer for occluded person re-identification.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|mask|pose|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10821:reviews/deep_16.md:2471:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10981:reviews/deep_23.md:664:misalignment between the visual and textual features caused by occlusions. These challenges motivate us
./reassess/r_3.md:11132:reviews/deep_6.md:1487:    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
./reassess/r_3.md:11148:reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11150:reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11152:reviews/deep_28.md:356:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11155:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11159:reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11161:reviews/deep_9.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11164:reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11166:reviews/deep_18.md:526:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11168:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11171:pivot/p_2.md:218:./reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11172:pivot/p_2.md:250:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11173:pivot/p_2.md:306:./reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11174:pivot/p_2.md:339:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11175:pivot/p_2.md:391:./reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11176:pivot/p_2.md:567:./reviews/deep_2.md:737:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11177:pivot/p_2.md:610:./reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11179:pivot/p_2.md:859:./reviews/deep_13.md:662:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11180:pivot/p_2.md:900:./reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11181:pivot/p_2.md:956:./reviews/deep_17.md:261:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11183:pivot/p_2.md:1078:./reviews/deep_12.md:431:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11184:pivot/p_2.md:1127:./reviews/deep_5.md:806:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11185:pivot/p_2.md:1176:./reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11186:pivot/p_2.md:1213:./reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11187:pivot/p_2.md:1308:./reviews/deep_20.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11188:pivot/p_2.md:1333:./reviews/deep_11.md:467:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11189:pivot/p_2.md:1396:./reviews/deep_24.md:839:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11190:pivot/p_2.md:1445:./reviews/deep_4.md:183:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11193:pivot/p_2.md:1506:./reviews/deep_10.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11194:pivot/p_2.md:1727:./reviews/deep_14.md:673:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11195:pivot/p_2.md:1801:./reviews/deep_3.md:556:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11196:pivot/p_2.md:1896:./reviews/deep_22.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11199:pivot/p_2.md:4077:./reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11200:pivot/p_2.md:4692:./reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11201:pivot/p_2.md:4731:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11202:pivot/p_2.md:4787:./reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11203:pivot/p_2.md:4820:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11204:pivot/p_2.md:4872:./reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11205:pivot/p_2.md:5176:reviews/deep_12.md:431:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11206:pivot/p_2.md:5191:reviews/deep_12.md:528:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11208:pivot/p_2.md:5243:reviews/deep_3.md:556:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11209:pivot/p_2.md:5293:reviews/deep_11.md:467:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11210:pivot/p_2.md:5308:reviews/deep_11.md:564:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11212:pivot/p_2.md:5398:reviews/deep_24.md:839:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11213:pivot/p_2.md:5413:reviews/deep_24.md:936:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11215:pivot/p_2.md:5452:reviews/deep_10.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11216:pivot/p_2.md:5467:reviews/deep_10.md:590:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11218:pivot/p_2.md:5515:reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11219:pivot/p_2.md:5590:reviews/deep_5.md:806:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11220:pivot/p_2.md:5605:reviews/deep_5.md:903:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11222:pivot/p_2.md:5672:pivot/p_2.md:218:./reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11223:pivot/p_2.md:5675:pivot/p_2.md:250:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11224:pivot/p_2.md:5677:pivot/p_2.md:306:./reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11225:pivot/p_2.md:5679:pivot/p_2.md:339:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11226:pivot/p_2.md:5682:pivot/p_2.md:391:./reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11227:pivot/p_2.md:5684:pivot/p_2.md:567:./reviews/deep_2.md:737:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11228:pivot/p_2.md:5690:pivot/p_2.md:610:./reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11230:pivot/p_2.md:5749:pivot/p_2.md:859:./reviews/deep_13.md:662:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11231:pivot/p_2.md:5751:pivot/p_2.md:900:./reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11232:pivot/p_2.md:5754:pivot/p_2.md:956:./reviews/deep_17.md:261:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11234:pivot/p_2.md:5789:pivot/p_2.md:1078:./reviews/deep_12.md:431:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11235:pivot/p_2.md:5791:pivot/p_2.md:1127:./reviews/deep_5.md:806:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11236:pivot/p_2.md:5795:pivot/p_2.md:1176:./reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11237:pivot/p_2.md:5798:pivot/p_2.md:1213:./reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11238:pivot/p_2.md:5846:pivot/p_2.md:1308:./reviews/deep_20.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11239:pivot/p_2.md:5848:pivot/p_2.md:1333:./reviews/deep_11.md:467:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11240:pivot/p_2.md:5855:pivot/p_2.md:1396:./reviews/deep_24.md:839:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11241:pivot/p_2.md:5857:pivot/p_2.md:1445:./reviews/deep_4.md:183:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11244:pivot/p_2.md:5869:pivot/p_2.md:1506:./reviews/deep_10.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11245:pivot/p_2.md:5886:pivot/p_2.md:1727:./reviews/deep_14.md:673:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11246:pivot/p_2.md:5894:pivot/p_2.md:1801:./reviews/deep_3.md:556:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11247:pivot/p_2.md:5899:pivot/p_2.md:1896:./reviews/deep_22.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11249:pivot/p_2.md:6217:pivot/p_2.md:4077:./reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11250:pivot/p_2.md:6331:pivot/p_2.md:4692:./reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11251:pivot/p_2.md:6335:pivot/p_2.md:4731:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11252:pivot/p_2.md:6337:pivot/p_2.md:4787:./reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11253:pivot/p_2.md:6339:pivot/p_2.md:4820:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11254:pivot/p_2.md:6342:pivot/p_2.md:4872:./reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11255:pivot/p_2.md:6363:reviews/deep_13.md:662:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11256:pivot/p_2.md:6378:reviews/deep_13.md:759:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11258:pivot/p_2.md:6454:reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11259:pivot/p_2.md:6469:reviews/deep_18.md:526:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11261:pivot/p_2.md:6546:reviews/deep_4.md:183:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11264:pivot/p_2.md:6673:reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11265:pivot/p_2.md:6688:reviews/deep_26.md:592:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11267:pivot/p_2.md:6719:reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11268:pivot/p_2.md:6734:reviews/deep_28.md:356:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11270:pivot/p_2.md:6830:reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11271:pivot/p_2.md:6845:reviews/deep_25.md:452:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11273:pivot/p_2.md:6913:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11275:pivot/p_2.md:6924:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11277:pivot/p_2.md:7050:reviews/deep_14.md:673:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11278:pivot/p_2.md:7065:reviews/deep_14.md:770:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11279:pivot/p_2.md:7115:reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11280:pivot/p_2.md:7130:reviews/deep_6.md:659:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11282:pivot/p_2.md:7186:reviews/deep_20.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11283:pivot/p_2.md:7201:reviews/deep_20.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11285:pivot/p_2.md:7308:reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11286:pivot/p_2.md:7323:reviews/deep_23.md:586:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11288:pivot/p_2.md:7409:reviews/deep_2.md:737:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11289:pivot/p_2.md:7424:reviews/deep_2.md:834:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11291:pivot/p_2.md:7478:reviews/deep_17.md:261:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11292:pivot/p_2.md:7493:reviews/deep_17.md:358:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11294:pivot/p_2.md:7549:reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11295:pivot/p_2.md:7564:reviews/deep_9.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11297:pivot/p_2.md:7599:reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11298:pivot/p_2.md:7614:reviews/deep_27.md:601:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11300:pivot/p_2.md:7773:reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11301:pivot/p_2.md:7788:reviews/deep_16.md:269:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11303:pivot/p_2.md:7856:reviews/deep_22.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11304:pivot/p_2.md:7871:reviews/deep_22.md:590:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11307:pivot/p_2.md:7972:analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11309:pivot/p_2.md:8154:analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11312:pivot/p_2.md:8294:./reviews/deep_2.md:737:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11313:pivot/p_2.md:8337:./reviews/deep_18.md:429:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11314:validate/v_3.md:676:./reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11315:validate/v_3.md:739:./reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11316:validate/v_3.md:749:./reviews/deep_23.md:586:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11318:validate/v_3.md:872:./reviews/deep_20.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11319:validate/v_3.md:882:./reviews/deep_20.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11322:validate/v_3.md:1032:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11323:validate/v_3.md:1042:./reviews/deep_28.md:356:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11325:validate/v_3.md:1291:./reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11326:validate/v_3.md:1301:./reviews/deep_16.md:269:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11328:validate/v_3.md:1369:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11329:validate/v_3.md:1379:./reviews/deep_9.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11331:validate/v_3.md:1848:./reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11332:validate/v_3.md:1858:./reviews/deep_25.md:452:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11335:validate/v_3.md:1887:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11337:validate/v_3.md:1906:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11341:validate/v_3.md:2945:../litreview2/analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11344:validate/v_3.md:3129:../litreview/reviews/lit_17.md:327:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11345:validate/v_3.md:3139:../litreview/reviews/lit_17.md:424:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11347:validate/v_3.md:3397:../litreview/reviews/lit_7.md:322:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11348:validate/v_3.md:3407:../litreview/reviews/lit_7.md:419:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11350:validate/v_3.md:3485:../litreview/reviews/lit_18.md:587:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11351:validate/v_3.md:3495:../litreview/reviews/lit_18.md:684:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11353:validate/v_3.md:3549:../litreview/reviews/lit_6.md:419:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11354:validate/v_3.md:3559:../litreview/reviews/lit_6.md:516:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11356:validate/v_3.md:4019:../litreview/reviews/lit_15.md:747:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11357:validate/v_3.md:4029:../litreview/reviews/lit_15.md:844:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11359:validate/v_3.md:4085:../litreview/reviews/lit_5.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11360:validate/v_3.md:4134:../litreview/reviews/lit_10.md:339:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11361:validate/v_3.md:4144:../litreview/reviews/lit_10.md:436:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11363:validate/v_3.md:4193:../litreview/reviews/lit_11.md:1578:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11364:validate/v_3.md:4203:../litreview/reviews/lit_11.md:1675:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11366:validate/v_3.md:4250:../litreview/reviews/lit_8.md:619:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11367:validate/v_3.md:4260:../litreview/reviews/lit_8.md:716:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11369:validate/v_3.md:4307:../litreview/reviews/lit_2.md:322:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11370:validate/v_3.md:4317:../litreview/reviews/lit_2.md:419:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11372:validate/v_3.md:4371:../litreview/reviews/lit_19.md:318:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11373:validate/v_3.md:4381:../litreview/reviews/lit_19.md:415:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11375:validate/v_3.md:4412:../litreview/reviews/lit_1.md:458:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11376:validate/v_3.md:4422:../litreview/reviews/lit_1.md:555:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11378:validate/v_3.md:4493:../litreview/reviews/lit_13.md:176:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11379:validate/v_3.md:4503:../litreview/reviews/lit_13.md:273:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11381:validate/v_3.md:4545:../litreview/reviews/lit_9.md:369:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11391:reviews/deep_2.md:737:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11393:reviews/deep_2.md:834:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11405:reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11407:reviews/deep_27.md:601:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11410:reviews/deep_4.md:183:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11412:reviews/deep_4.md:696:module that integrates visual, textual, and crossattention features based on their reliability. Experimental results demonstrate that FLaN-Net outperforms existing methods on occluded ReID benchmarks, offering a robust solution for challenging
./reassess/r_3.md:11414:reviews/deep_4.md:731:proposed FLaN-Net method, which incorporates fine-grained textual descriptions and a more effective multi-modal interaction module, enables the model to achieve more robust noise suppression.
./reassess/r_3.md:11418:reviews/deep_4.md:790:encapsulates the essential identity of the pedestrian. The attribute token A∗ and the context token O∗ serve as detail tokens, capturing identity-relevant attributes and occluding objects respectively. FLaN-Net learns detailed information for
./reassess/r_3.md:11419:reviews/deep_4.md:793:prompts, FLaN-Net incorporates a cross-attention mechanism that dynamically aligns textual tokens with image patch
./reassess/r_3.md:11458:reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11460:reviews/deep_16.md:269:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11464:reviews/deep_13.md:662:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11466:reviews/deep_13.md:759:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11471:reviews/deep_22.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11473:reviews/deep_22.md:590:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11477:reviews/deep_6.md:562:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11479:reviews/deep_6.md:659:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11483:reviews/deep_26.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11485:reviews/deep_26.md:592:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11492:validate/v_2.md:3169:./reviews/deep_4.md:793:prompts, FLaN-Net incorporates a cross-attention mechanism that dynamically aligns textual tokens with image patch
./reassess/r_3.md:11498:validate/v_2.md:5109:./reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11499:validate/v_2.md:5121:./reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11500:validate/v_2.md:5173:./analyses/deep_25.txt:67:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11502:validate/v_2.md:5932:./analyses/ALL_ANALYSES.txt:1184:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11503:validate/v_2.md:8056:validate/v_2.md:3169:./reviews/deep_4.md:793:prompts, FLaN-Net incorporates a cross-attention mechanism that dynamically aligns textual tokens with image patch
./reassess/r_3.md:11504:validate/v_2.md:10165:reviews/deep_4.md:793:prompts, FLaN-Net incorporates a cross-attention mechanism that dynamically aligns textual tokens with image patch
./reassess/r_3.md:11505:validate/v_2.md:12513:validate/v_3.md:676:./reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11506:validate/v_2.md:12516:validate/v_3.md:739:./reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11507:validate/v_2.md:12528:validate/v_3.md:872:./reviews/deep_20.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11508:validate/v_2.md:12545:validate/v_3.md:1032:./reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11509:validate/v_2.md:12574:validate/v_3.md:1291:./reviews/deep_16.md:172:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11510:validate/v_2.md:12578:validate/v_3.md:1369:./reviews/deep_9.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11511:validate/v_2.md:12647:validate/v_3.md:1848:./reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11512:validate/v_2.md:12749:validate/v_3.md:3129:../litreview/reviews/lit_17.md:327:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11513:validate/v_2.md:12783:validate/v_3.md:3397:../litreview/reviews/lit_7.md:322:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11514:validate/v_2.md:12799:validate/v_3.md:3485:../litreview/reviews/lit_18.md:587:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11515:validate/v_2.md:12805:validate/v_3.md:3549:../litreview/reviews/lit_6.md:419:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11516:validate/v_2.md:12859:validate/v_3.md:4019:../litreview/reviews/lit_15.md:747:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11517:validate/v_2.md:12875:validate/v_3.md:4085:../litreview/reviews/lit_5.md:495:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11518:validate/v_2.md:12879:validate/v_3.md:4134:../litreview/reviews/lit_10.md:339:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11519:validate/v_2.md:12883:validate/v_3.md:4193:../litreview/reviews/lit_11.md:1578:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11520:validate/v_2.md:12893:validate/v_3.md:4250:../litreview/reviews/lit_8.md:619:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11521:validate/v_2.md:12899:validate/v_3.md:4307:../litreview/reviews/lit_2.md:322:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11522:validate/v_2.md:12910:validate/v_3.md:4371:../litreview/reviews/lit_19.md:318:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11523:validate/v_2.md:12920:validate/v_3.md:4412:../litreview/reviews/lit_1.md:458:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11524:validate/v_2.md:12928:validate/v_3.md:4493:../litreview/reviews/lit_13.md:176:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11525:validate/v_2.md:12932:validate/v_3.md:4545:../litreview/reviews/lit_9.md:369:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11531:reviews/deep_17.md:261:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11533:reviews/deep_17.md:358:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11540:reviews/deep_11.md:467:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11542:reviews/deep_11.md:564:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11545:reviews/deep_20.md:823:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11547:reviews/deep_20.md:920:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11553:reviews/deep_3.md:556:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11557:reviews/deep_25.md:355:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11559:reviews/deep_25.md:452:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11575:reviews/deep_25.md:6852:non-targets. As the appearance of target and non-target pedestrians often exhibits notable texture variations, it is more feasible to leverage
./reassess/r_3.md:11613:reviews/deep_25.md:8155:texture-aware encoder (left column), TTPM accurately re-identiﬁes the
./reassess/r_3.md:11615:reviews/deep_25.md:8160:textures. This result indicates that the proposed TTPM still has limitations in relying on texture features. While these features enable accurate
./reassess/r_3.md:11623:reviews/deep_25.md:12347:TTPM 的页面排版导致方法文本顺序有些乱，刚才截取漏了 MFE 的前半段。我会再补一次该论文从 3.1 到 3.3 的开头，避免把 pose-patch mapping 的逻辑链断掉。
./reassess/r_3.md:11630:reviews/deep_25.md:14097:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11632:reviews/deep_25.md:14176:2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。
./reassess/r_3.md:11635:reviews/deep_5.md:806:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11637:reviews/deep_5.md:903:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11643:reviews/deep_23.md:489:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11645:reviews/deep_23.md:586:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11660:reviews/deep_10.md:493:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11662:reviews/deep_10.md:590:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11677:reviews/deep_14.md:673:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11679:reviews/deep_14.md:770:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11686:reviews/deep_14.md:2372:   244	image–text matching task. Human-centric models like UniHCP, PATH, and SOLIDER demonstrate the potential of
./reassess/r_3.md:11694:reviews/deep_24.md:839:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11696:reviews/deep_24.md:936:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11712:reviews/deep_12.md:431:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reassess/r_3.md:11714:reviews/deep_12.md:528:- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
./reassess/r_3.md:11723:web search: texture aware transformer pose patch mapping occluded person re identification Pattern Recognition 2025 TTPM
./reassess/r_3.md:11737:结论：**值得测，但只值得做一次零训练 kill-switch；不值得直接开训练。**  
./reassess/r_3.md:11738:先验很低，因为它已经撞到 FED 的 NTP feature contamination、KPR 的 multi-person ambiguity、QPM 的 target-pixel attention、DPEFormer 的 occlusion-free token selection，以及 TTPM 那条 non-target pedestrian occlusion/texture-aware 路线。能活下来的唯一差异是：**不讲“遮挡区域噪声”，只讲“遮挡者身份向量把 target embedding 拉向 donor-ID，并且可被无身份信息的 source separation 压掉后直接涨 ReID”。**
./reassess/r_3.md:11748:**最锋利 kill-switch：合成 NTP donor 注入 + 三种压制臂**
./reassess/r_3.md:11785:我的判断：**可以测，成本低，24 小时内能判生死；但大概率会死在“泄漏存在但压了不涨”或“mask-gate 等价”上。** 如果这个 kill-switch 不过，遮挡这块就应彻底关，不要再用 SMPL、visibility、completion、occluder gate、source separation 换壳。当前环境只读，我没有落盘 design 或启动实验。
./reassess/r_3.md:11788:结论：**值得测，但只值得做一次零训练 kill-switch；不值得直接开训练。**  
./reassess/r_3.md:11789:先验很低，因为它已经撞到 FED 的 NTP feature contamination、KPR 的 multi-person ambiguity、QPM 的 target-pixel attention、DPEFormer 的 occlusion-free token selection，以及 TTPM 那条 non-target pedestrian occlusion/texture-aware 路线。能活下来的唯一差异是：**不讲“遮挡区域噪声”，只讲“遮挡者身份向量把 target embedding 拉向 donor-ID，并且可被无身份信息的 source separation 压掉后直接涨 ReID”。**
./reassess/r_3.md:11799:**最锋利 kill-switch：合成 NTP donor 注入 + 三种压制臂**
./reassess/r_3.md:11836:我的判断：**可以测，成本低，24 小时内能判生死；但大概率会死在“泄漏存在但压了不涨”或“mask-gate 等价”上。** 如果这个 kill-switch 不过，遮挡这块就应彻底关，不要再用 SMPL、visibility、completion、occluder gate、source separation 换壳。当前环境只读，我没有落盘 design 或启动实验。

codex
本地 `rg` 输出很大，但有两点可用：此前已经判断“语义/VLM teacher 直接补遮挡”对强 SOLIDER 偏负；RSTPReid 的新价值不能来自泛泛语义 teacher，而要来自文本标注本身提供的可测错误变量。
exec
/bin/zsh -lc "sed -n '1,220p' pivot/p_1.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef8b9-edb6-7b81-bdfa-2a152fca61b2
--------
user
背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。

**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
- ①σ_aerial < σ_ground(航拍q156.96/g167.47 < 地面q171.64/g172.81双侧; 合成退化σ反而降115.41<118.93)。'航拍更欠定=宽分布'前提**错的**——航拍低清=少高频细节=更平滑=低TTA方差。
- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
- ③image-level σ无用(C3 view-mean 69.07不降反升/C4同视角置换67.47/C5 hardness 66.63 都≈correct不掉)。

**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。

**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。

== 你的任务 ==
角色=**救援者**。给定新发现 σ_aerial<σ_ground(航拍更平滑/少细节, 地面细节丰富)+ avg>MaxSim 这两个硬事实, 用联网搜索找一个**还活着的 B 类 re-frame**, 必须同时满足: (a)符合'航拍平滑低细节'而非'航拍噪声/欠定'; (b)不 reduce to avg(机制不能退化成平均池化); (c)有廉价零训练 kill-switch; (d)不撞 GSAlign/VDT/SeCap/ViSA/cross-resolution 红海。逐个候选查 novelty。如果找不到符合的, 诚实说'救不动'。务实中文, 给候选+kill-switch+撞车核查。
codex
我会先核对近年航拍-地面 ReID、跨分辨率 ReID、token/patch 聚合与池化相关方法，重点看是否已经把“航拍平滑低细节”当作核心问题来建模。随后按你的四个约束筛候选，并给每个候选配一个零训练 kill-switch。
web search: 
web search: aerial ground person re-identification AG-ReID CARGO GSAlign VDT SeCap ViSA paper
web search: 
web search: GSAlign aerial ground person re-identification paper
web search: 
web search: person re-identification token pooling average pooling max pooling transformer global average pooling patch tokens local detail ReID 2024
web search: 
web search: person re-identification frequency domain low high frequency texture detail 2024 paper
codex
初步检索确认几个红海边界：VDT/ViSA/SeCap/GSAlign都把核心矛盾写成视角、语义/局部对齐、几何错位或 view-aware token/expert；频域思路在 VI-ReID 和跨分辨率 ReID 里很多，但在 AG-ReID 里还没看到把“航拍更平滑、地面细节过剩”作为主问题来立论。下一步我会专门查“低频/纹理/细节过剩/shortcut/augmentation”一类路线，避免把候选变成普通跨分辨率。
web search: 
web search: person re-identification high frequency shortcut texture bias low frequency representation paper
web search: 
web search: AGPReID aerial ground person re-identification 2026 arxiv view-aware semantic alignment ViSA
web search: 
web search: cross-resolution person re-identification high frequency low frequency suppression detail paper
web search: 
web search: "Frequency" "Person Re-identification" "ReID" "CVPR"
web search: 
web search: person re-identification covariance pooling second-order statistics feature aggregation
web search: 
web search: person re-identification style statistics mean variance feature "style" "ReID"
web search: 
web search: "semantic granularity" "person re-identification"
web search: 
web search: person re-identification optimal transport token matching local feature similarity coverage
web search: 
web search: https://arxiv.org/abs/2510.22268
web search: 
web search: https://arxiv.org/abs/2403.14513
ERROR: Reconnecting... 1/5
web search: 
web search: aerial-ground person re-identification VDT view decoupling transformer arxiv
web search: 
web search: "ViSA" "aerial-ground" "person re-identification"
web search: 
web search: cross resolution person re-identification survey high resolution low resolution super resolution degradation invariant methods 2024
web search: 
web search: "Frequency Domain" "Visible-Infrared Person Re-Identification"
web search: 
web search: person re-identification DCT feature pooling low frequency spatial harmonics
web search: 
web search: "DCT" "ReID" "person" "frequency"
web search: 
web search: AG-ReID.v2 aerial ground person re-identification dataset paper
web search: 
web search: "frequency" "domain generalization" "person re-identification"
web search: 
web search: "Person Re-Identification" "Fourier" "domain generalization"
codex
**结论**

还没到“救不动”。我认为唯一相对还活着的 B 类 re-frame 是：

**航拍-地面 ReID 不是“低清欠定”或“局部对齐失败”，而是“共同可支持的空间带宽不对称”问题。**  
航拍图像更平滑，说明它稳定传递的是低阶空间结构；地面图像多出来的细节不是补充证据，而是 A-G 匹配中没有航拍支持的高频私有证据。`avg > MaxSim` 说明局部最强 patch 证据反而危险，但 `avg` 只是 DC-only 近似，下一步应验证 **低阶非 DC 空间谐波** 是否比纯平均更有身份信息。

**候选 1：SHSM / Smooth Harmonic Support Matching**

核心机制：

- 从 Swin token map 做 2D DCT / Fourier 低阶投影。
- `DC` 系数等价于 global average，是强对照。
- 保留 `low non-DC` 系数，例如头-躯干-腿的粗空间布局、上下/左右颜色分布。
- 丢弃或隔离 mid/high frequency 地面细节。
- 训练端可做 fixed harmonic pooling + high-band dropout/swap，但主贡献必须落在“低阶非 DC 支架”，不是 blur、SR、avg pooling。

为什么符合两个硬事实：

- `σ_aerial < σ_ground`：航拍不是噪声更大，而是高频少、低阶结构更稳定。
- `avg > MaxSim`：错误不是没有局部对齐，而是局部高频/patch max 在 A-G 中抓到伪证据。
- `avg` 的成功可以被解释为“退到 DC 避免高频伤害”；但如果 `low non-DC` 能进一步涨，才证明不是 reduce to avg。

零训练 kill-switch：

```text
冻结 swin_fix256，取最后 token map F[H,W,C]
对 H,W 做 2D DCT
D0   = C[0,0,:]                      # DC = avg
Dlow = low non-DC coefficients         # 如 u+v<=2 且非(0,0)
Dhi  = mid/high coefficients
```

必须跑这些对照：

1. `D0` vs `D0 + Dlow`  
   如果 `D0 + Dlow` 不能稳定高于 `D0`，直接判死，因为它退化成 avg。

2. `Dlow` vs `Dhi`  
   如果 high-band 比 low non-DC 更有用，重定义不成立。

3. `D0 + Dlow + Dhi` vs `D0 + Dlow`  
   如果加 high-band 提升 A→G，说明“地面细节是私有伪证据”不成立。

4. band-level TTA σ  
   复查 `σ_aerial < σ_ground` 是否主要发生在 mid/high band；如果低阶 band 也乱，故事不干净。

通过门槛建议很硬一点：  
`D0 + Dlow` 至少在 CARGO A→G 比 `D0` 高 `+0.5 mAP` 以上，并且 AG-ReID.v2 不掉；否则不要开训练。

**撞车核查**

- 不撞 GSAlign：GSAlign 是 TPS 几何变形 + semantic/visibility mask，对齐 severe pose / spatial misalignment。SHSM 不做 keypoint、warp、visibility mask。来源：[GSAlign arXiv](https://arxiv.org/abs/2510.22268)
- 不撞 VDT：VDT 是 view-related / view-unrelated decoupling。SHSM 不是 view expert，也不是正交解耦，而是可观测空间带宽。来源：[VDT arXiv](https://arxiv.org/abs/2403.14513)
- 不撞 SeCap：SeCap 是 adaptive prompt + local feature refinement。SHSM 反而反对局部细节主导。来源：[SeCap arXiv](https://arxiv.org/abs/2503.06965)
- 不撞 ViSA：ViSA 是 view-aware semantic queries + graph local fusion。SHSM 不做 semantic local alignment。来源：[ViSA arXiv](https://arxiv.org/abs/2605.18192)
- 有跨分辨率撞车风险，但可避开：CR-ReID 常见是 SR、multi-resolution fusion、resolution-invariant distillation、feature panning。SHSM 不能写成“低清到高清”或“resolution gap”，必须写成 **A-G 方向性的 unsupported high-frequency evidence**。参考红线：[RFD](https://arxiv.org/abs/2109.07871)、[MRJL](https://arxiv.org/abs/2105.12684)、[VPFA](https://arxiv.org/abs/2510.00936)
- 频域本身不新，VI-ReID 已有频域方法，所以 novelty 不能是“用了 DCT/Fourier”。只能是：`avg` 是 DC-only 症状，AG-ReID 需要低阶非 DC harmonic scaffold。参考风险：[FDMNet](https://arxiv.org/abs/2401.01839)

**其他候选判定**

- 纯 ground blur / low-pass：不合格。太像 cross-resolution / degradation robustness，也容易 reduce to avg。
- coverage/consensus token matching：可做辅助 kill-switch，但单独做主线会撞 local matching / ViSA / OT 类方法。
- style mean-variance / Gram pooling：不够 B，ReID/style/second-order 旧味太重，除非零训练大幅超过 avg，否则不建议。

最终建议：先跑 **DCT token-band kill-switch**。如果 `low non-DC` 不能赢 `DC(avg)`，这条也死；那就基本可以诚实说当前两个硬事实救不出干净 B 类方法稿。
tokens used
31,595
**结论**

还没到“救不动”。我认为唯一相对还活着的 B 类 re-frame 是：

**航拍-地面 ReID 不是“低清欠定”或“局部对齐失败”，而是“共同可支持的空间带宽不对称”问题。**  
航拍图像更平滑，说明它稳定传递的是低阶空间结构；地面图像多出来的细节不是补充证据，而是 A-G 匹配中没有航拍支持的高频私有证据。`avg > MaxSim` 说明局部最强 patch 证据反而危险，但 `avg` 只是 DC-only 近似，下一步应验证 **低阶非 DC 空间谐波** 是否比纯平均更有身份信息。

**候选 1：SHSM / Smooth Harmonic Support Matching**

核心机制：

- 从 Swin token map 做 2D DCT / Fourier 低阶投影。
- `DC` 系数等价于 global average，是强对照。
- 保留 `low non-DC` 系数，例如头-躯干-腿的粗空间布局、上下/左右颜色分布。
- 丢弃或隔离 mid/high frequency 地面细节。
- 训练端可做 fixed harmonic pooling + high-band dropout/swap，但主贡献必须落在“低阶非 DC 支架”，不是 blur、SR、avg pooling。

为什么符合两个硬事实：

- `σ_aerial < σ_ground`：航拍不是噪声更大，而是高频少、低阶结构更稳定。
- `avg > MaxSim`：错误不是没有局部对齐，而是局部高频/patch max 在 A-G 中抓到伪证据。
- `avg` 的成功可以被解释为“退到 DC 避免高频伤害”；但如果 `low non-DC` 能进一步涨，才证明不是 reduce to avg。

零训练 kill-switch：

```text
冻结 swin_fix256，取最后 token map F[H,W,C]
对 H,W 做 2D DCT
D0   = C[0,0,:]                      # DC = avg
Dlow = low non-DC coefficients         # 如 u+v<=2 且非(0,0)
Dhi  = mid/high coefficients
```

必须跑这些对照：

1. `D0` vs `D0 + Dlow`  
   如果 `D0 + Dlow` 不能稳定高于 `D0`，直接判死，因为它退化成 avg。

2. `Dlow` vs `Dhi`  
   如果 high-band 比 low non-DC 更有用，重定义不成立。

3. `D0 + Dlow + Dhi` vs `D0 + Dlow`  
   如果加 high-band 提升 A→G，说明“地面细节是私有伪证据”不成立。

4. band-level TTA σ  
   复查 `σ_aerial < σ_ground` 是否主要发生在 mid/high band；如果低阶 band 也乱，故事不干净。

通过门槛建议很硬一点：  
`D0 + Dlow` 至少在 CARGO A→G 比 `D0` 高 `+0.5 mAP` 以上，并且 AG-ReID.v2 不掉；否则不要开训练。

**撞车核查**

- 不撞 GSAlign：GSAlign 是 TPS 几何变形 + semantic/visibility mask，对齐 severe pose / spatial misalignment。SHSM 不做 keypoint、warp、visibility mask。来源：[GSAlign arXiv](https://arxiv.org/abs/2510.22268)
- 不撞 VDT：VDT 是 view-related / view-unrelated decoupling。SHSM 不是 view expert，也不是正交解耦，而是可观测空间带宽。来源：[VDT arXiv](https://arxiv.org/abs/2403.14513)
- 不撞 SeCap：SeCap 是 adaptive prompt + local feature refinement。SHSM 反而反对局部细节主导。来源：[SeCap arXiv](https://arxiv.org/abs/2503.06965)
- 不撞 ViSA：ViSA 是 view-aware semantic queries + graph local fusion。SHSM 不做 semantic local alignment。来源：[ViSA arXiv](https://arxiv.org/abs/2605.18192)
- 有跨分辨率撞车风险，但可避开：CR-ReID 常见是 SR、multi-resolution fusion、resolution-invariant distillation、feature panning。SHSM 不能写成“低清到高清”或“resolution gap”，必须写成 **A-G 方向性的 unsupported high-frequency evidence**。参考红线：[RFD](https://arxiv.org/abs/2109.07871)、[MRJL](https://arxiv.org/abs/2105.12684)、[VPFA](https://arxiv.org/abs/2510.00936)
- 频域本身不新，VI-ReID 已有频域方法，所以 novelty 不能是“用了 DCT/Fourier”。只能是：`avg` 是 DC-only 症状，AG-ReID 需要低阶非 DC harmonic scaffold。参考风险：[FDMNet](https://arxiv.org/abs/2401.01839)

**其他候选判定**

- 纯 ground blur / low-pass：不合格。太像 cross-resolution / degradation robustness，也容易 reduce to avg。
- coverage/consensus token matching：可做辅助 kill-switch，但单独做主线会撞 local matching / ViSA / OT 类方法。
- style mean-variance / Gram pooling：不够 B，ReID/style/second-order 旧味太重，除非零训练大幅超过 avg，否则不建议。

最终建议：先跑 **DCT token-band kill-switch**。如果 `low non-DC` 不能赢 `DC(avg)`，这条也死；那就基本可以诚实说当前两个硬事实救不出干净 B 类方法稿。

exec
/bin/zsh -lc "sed -n '11730,11820p' reassess/r_3.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
web search: 
web search: ProFD occluded person re-identification prompt learning
web search: 
web search: https://arxiv.org/abs/2407.18112
web search: 
web search: https://arxiv.org/abs/2112.08740
codex
结论：**值得测，但只值得做一次零训练 kill-switch；不值得直接开训练。**  
先验很低，因为它已经撞到 FED 的 NTP feature contamination、KPR 的 multi-person ambiguity、QPM 的 target-pixel attention、DPEFormer 的 occlusion-free token selection，以及 TTPM 那条 non-target pedestrian occlusion/texture-aware 路线。能活下来的唯一差异是：**不讲“遮挡区域噪声”，只讲“遮挡者身份向量把 target embedding 拉向 donor-ID，并且可被无身份信息的 source separation 压掉后直接涨 ReID”。**

我查到的关键先例：  
[FED](https://arxiv.org/abs/2112.08740) 已明确把遮挡拆成 NPO 和 NTP，并说 NTP 会 feature contamination。  
[KPR](https://arxiv.org/abs/2407.18112) 明确处理 multi-person ambiguity，指出多人体 bbox 会 feature mixing 或关注错人。  
[QPM](https://arxiv.org/abs/2201.00107) 做 target pedestrian identity-aware spatial attention，并按 common non-occluded regions 匹配。  
[DPEFormer](https://arxiv.org/abs/2402.10435) 做动态 occlusion-free patch token selection。  
[DPL-ReID](https://arxiv.org/abs/2605.19527) 也已经开始说 occluder semantic information。  
所以 novelty 只剩很窄的“**donor-ID leakage 可测、可减、减了涨**”。

**最锋利 kill-switch：合成 NTP donor 注入 + 三种压制臂**
用强 ckpt `exp255` 冻结，sanity 先复现 `~73.05 mAP`。不训练。

构造数据：  
从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。

评估 5 条：

1. `clean`: 原始 `T` 查 gallery。  
2. `mixed`: 合成 `M` 查 gallery，确认遮挡造成掉点。  
3. `mask-gate`: 用真实 donor mask 直接去掉/不池化 donor 区域 token。回答“普通 occluder gate 上界”。  
4. `oracle-donor-sub`: 用 donor 原图特征 `f_D` 做 `normalize(f_M - λ proj_{f_D}(f_M))`，扫 `λ`。这是身份 oracle 上界。  
5. `region-sub`: 只用 `M` 里 donor mask 区域的 pooled feature `r_occ` 做 `normalize(f_M - λ proj_{r_occ}(f_M))`。这是**无 donor-ID 可实现版上界**。

必须同时报两个指标：

- 泄漏存在性：`cos(f_M, f_D)-cos(f_T, f_D)`、donor pid 进入 top-k 的比例、donor rank 是否显著提前。
- ReID 真实收益：mAP/R1 是否从 `mixed` 回升，回收 clean→mixed 掉点的百分比。

**死亡判据**
满足任一条就关遮挡：

- donor leakage 存在，但 `region-sub` 不涨：说明污染可测但不可用，撞 exp109 墙。
- `oracle-donor-sub` 都不涨 `+1.0 mAP` 或回收不到 `20%` clean→mixed 掉点：source-separation 本体无 headroom。
- `mask-gate` 涨，`region-sub/oracle-sub` 不比 mask-gate 强：只是旧 occluder-gate/visibility 换名。
- 随机 donor 向量、随机 mask、object occluder 控制也涨：检索 artifact。
- 收益只在合成图有，真实 multi-person query 子集无 `+1 mAP`：不可投稿。

**通过判据**
只有这样才继续：

- `mixed` 明确掉点，比如 `-5 mAP` 量级；
- donor pid top-k 明显上升，证明真有身份泄漏；
- `oracle-donor-sub` 至少 `+1.5 mAP`，回收 `>30%` 掉点；
- `region-sub` 至少 `+0.8~1.0 mAP`，且显著强于 mask-gate/random 控制；
- 真实 multi-person query 子集也涨，且 clean query 不伤。

我的判断：**可以测，成本低，24 小时内能判生死；但大概率会死在“泄漏存在但压了不涨”或“mask-gate 等价”上。** 如果这个 kill-switch 不过，遮挡这块就应彻底关，不要再用 SMPL、visibility、completion、occluder gate、source separation 换壳。当前环境只读，我没有落盘 design 或启动实验。
tokens used
238,528
结论：**值得测，但只值得做一次零训练 kill-switch；不值得直接开训练。**  
先验很低，因为它已经撞到 FED 的 NTP feature contamination、KPR 的 multi-person ambiguity、QPM 的 target-pixel attention、DPEFormer 的 occlusion-free token selection，以及 TTPM 那条 non-target pedestrian occlusion/texture-aware 路线。能活下来的唯一差异是：**不讲“遮挡区域噪声”，只讲“遮挡者身份向量把 target embedding 拉向 donor-ID，并且可被无身份信息的 source separation 压掉后直接涨 ReID”。**

我查到的关键先例：  
[FED](https://arxiv.org/abs/2112.08740) 已明确把遮挡拆成 NPO 和 NTP，并说 NTP 会 feature contamination。  
[KPR](https://arxiv.org/abs/2407.18112) 明确处理 multi-person ambiguity，指出多人体 bbox 会 feature mixing 或关注错人。  
[QPM](https://arxiv.org/abs/2201.00107) 做 target pedestrian identity-aware spatial attention，并按 common non-occluded regions 匹配。  
[DPEFormer](https://arxiv.org/abs/2402.10435) 做动态 occlusion-free patch token selection。  
[DPL-ReID](https://arxiv.org/abs/2605.19527) 也已经开始说 occluder semantic information。  
所以 novelty 只剩很窄的“**donor-ID leakage 可测、可减、减了涨**”。

**最锋利 kill-switch：合成 NTP donor 注入 + 三种压制臂**
用强 ckpt `exp255` 冻结，sanity 先复现 `~73.05 mAP`。不训练。

构造数据：  
从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。

评估 5 条：

1. `clean`: 原始 `T` 查 gallery。  
2. `mixed`: 合成 `M` 查 gallery，确认遮挡造成掉点。  
3. `mask-gate`: 用真实 donor mask 直接去掉/不池化 donor 区域 token。回答“普通 occluder gate 上界”。  
4. `oracle-donor-sub`: 用 donor 原图特征 `f_D` 做 `normalize(f_M - λ proj_{f_D}(f_M))`，扫 `λ`。这是身份 oracle 上界。  
5. `region-sub`: 只用 `M` 里 donor mask 区域的 pooled feature `r_occ` 做 `normalize(f_M - λ proj_{r_occ}(f_M))`。这是**无 donor-ID 可实现版上界**。

必须同时报两个指标：

- 泄漏存在性：`cos(f_M, f_D)-cos(f_T, f_D)`、donor pid 进入 top-k 的比例、donor rank 是否显著提前。
- ReID 真实收益：mAP/R1 是否从 `mixed` 回升，回收 clean→mixed 掉点的百分比。

**死亡判据**
满足任一条就关遮挡：


exec
/bin/zsh -lc "sed -n '1,220p' pivot/p_2.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef8b9-edea-79d0-84a8-2c7a3b2a95ba
--------
user
背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。

**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
- ①σ_aerial < σ_ground(航拍q156.96/g167.47 < 地面q171.64/g172.81双侧; 合成退化σ反而降115.41<118.93)。'航拍更欠定=宽分布'前提**错的**——航拍低清=少高频细节=更平滑=低TTA方差。
- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
- ③image-level σ无用(C3 view-mean 69.07不降反升/C4同视角置换67.47/C5 hardness 66.63 都≈correct不掉)。

**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。

**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。

== 你的任务 ==
角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。
codex
我会先读本地实验脉络，确认哪些路线已经被文档明确判负；同时联网做撞车核查，避免把已有的几何对齐/可见性路线换个名字再提。
exec
/bin/zsh -lc "pwd && rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md' -g 'design.md' -g 'monitor.md' -g '*exp333*' -g 'AGENTS.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2

exec
/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
./strategic_pivot.sh:2:# 3-codex 战略 panel: B containment FAIL 后, 救援/转向/残酷否决, --search
./strategic_pivot.sh:8:CTX="背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。
./strategic_pivot.sh:10:**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
./strategic_pivot.sh:12:- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
./strategic_pivot.sh:15:**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。
./strategic_pivot.sh:17:**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。"
./strategic_pivot.sh:20:ROLES[1]="角色=**救援者**。给定新发现 σ_aerial<σ_ground(航拍更平滑/少细节, 地面细节丰富)+ avg>MaxSim 这两个硬事实, 用联网搜索找一个**还活着的 B 类 re-frame**, 必须同时满足: (a)符合'航拍平滑低细节'而非'航拍噪声/欠定'; (b)不 reduce to avg(机制不能退化成平均池化); (c)有廉价零训练 kill-switch; (d)不撞 GSAlign/VDT/SeCap/ViSA/cross-resolution 红海。逐个候选查 novelty。如果找不到符合的, 诚实说'救不动'。务实中文, 给候选+kill-switch+撞车核查。"
./strategic_pivot.sh:21:ROLES[2]="角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。"
./strategic_pivot.sh:22:ROLES[3]="角色=**残酷否决者**。不留情面判断: 整个 CARGO/aerial-ground empirical 方向, 对一篇 B 类**方法稿**(不是 benchmark/不是 backbone), 是不是已经是死胡同?证据: 遮挡5死角→转empirical→OVLI/OVP/MaxSim/containment 又全倒, method 始终不成形, 只有 avg 这个平凡 baseline 撑着。联网核查 aerial-ground ReID 2024-2026 现状(还有没有 method 空间, 还是已被 GSAlign/VDT/AG-VPReID 这代占满)。如果是死胡同, 明确说'该放弃换战场', 并指出团队最该回到哪类问题(基于它的真实资产: SMPL几何/Swin/遮挡历史)。如果不是死胡同, 指出唯一还值得赌的点。务实中文, 别和稀泥。"
./analyses/deep_12.txt:48:**触发观察：** 无监督可见光红外 ReID 里，大家用对比学习拉近跨模态同类、推远异类。但作者指出，由于跨模态同类间隔很大，中间常常夹着异类样本。对比学习推远这些异类时，会阻碍真正同类跨模态样本靠近。
./cvpb_containment_killswitch_design.md:1:# 候选 B 零训练 kill-switch 设计（cvpb_containment_killswitch.py）
./cvpb_containment_killswitch_design.md:4:检验 re-frame："航拍-地面 ReID 不是对称匹配，而是物理定向的不确定性包含"。
./cvpb_containment_killswitch_design.md:11:3. 收益来自图像级非对称包含，不是混杂：8 个破坏对照全部掉分。
./cvpb_containment_killswitch_design.md:16:- 包含距离(A→G, query 航拍 a, gallery 地面 g)：D = KL(N_g‖N_a)，升序检索。
./cvpb_containment_killswitch_design.md:28:按航拍 query bbox 面积(area_of=h·w)分 4 桶(CARGO 无 A0/A1/A2 altitude 文件夹，故用面积)，报每桶 cosine vs 正确包含 mAP，预期最小面积桶包含收益最大。
./cvpb_containment_killswitch_design.md:40:true-pair 包含距离 << impostor。全过 = 隐藏变量证实；任一关键条不过 = B 降级。
./cvpb_containment_killswitch_design.md:79:2. **假设2 部分**: KL(g‖a) A→G=68.62 仅 +1.2 over cosine, 但 equal-var Maha(σ全常量)=67.94 同样 > cosine → "增益"来自 σ-free 距离形式, 不是包含。最佳对称分布距离(sym-KL/JS/Bhatt)全 << cosine。
./cvpb_containment_killswitch_design.md:81:- **方向不对称是检索 artifact**: 真正起作用的是"均值项除以 query 端方差"(A→G query=航拍→好; G→A query=地面→好), 除以 gallery 端方差必崩(高方差 gallery 永远排前)。G→A CORRECT(除 gallery-aerial σ)崩到 17.37 = 铁证。这不是物理包含, 是分母选谁的归一化效应。
./cvpb_containment_full.log:1:=> CARGO loaded (protocol-1 ALL)
./cvpb_containment_full.log:384:  (expectation: SMALLEST area bucket -> largest containment gain)
./cvpb_containment_full.log:396:[done] kill-switch complete. See B_CONTAINMENT_DESIGN.md sec.4 pass criteria.
./SYNTHESIS_METHODOLOGY.md:27:| 10 | **非对称包含** | PDA: 文本分布⊇图像分布; 方差=语义范围(非噪声) |
./SYNTHESIS_METHODOLOGY.md:29:| 12 | **表示形态错了** | 3D Gait: 稀疏SMPL参数没法和稠密appearance融→蒸成稠密时空场; "换辅助模态"消融证 dense>skeleton>SMPL参 |
./SYNTHESIS_METHODOLOGY.md:43:## 三、应用到我们自己(候选 re-framing, 每个带廉价 kill-switch)
./SYNTHESIS_METHODOLOGY.md:45:我们独有资产: CARGO/AG-ReID.v2 极端跨视角(航拍↔地面 90°, 航拍低清俯视); **观察: avg-pool 52.37 > token-MaxSim 45.19(差7分)**; SMPL 几何基建; Swin 67.33(backbone)。
./SYNTHESIS_METHODOLOGY.md:47:> ⚠️ 用户已打掉"MaxSim<avg→局部不可靠→用avg"(推到底=零贡献)。**新候选的硬门槛: 机制必须 beat avg, 不能退化成 avg。** aerial-ground 几何/可见性是红海(避开)。
./SYNTHESIS_METHODOLOGY.md:49:### 候选 B(主推): 非对称包含 — 把对称匹配换成"航拍证据⊆地面证据"
./SYNTHESIS_METHODOLOGY.md:50:- **观察**: 航拍低清俯视 = 信息**欠定**(看不清脸/纹理); 地面高清正面 = 信息**确定**。但所有 cross-view ReID 用**对称** cosine/MaxSim 匹配, 默认两视角信息对等。
./SYNTHESIS_METHODOLOGY.md:51:- **重定义(move 10+22)**: "大家以为跨视角是对称对齐问题, 其实航拍↔地面是**非对称信息包含**: 航拍身份证据是一个**范围(宽分布)**, 应被地面的**窄分布包含**。" 方差=该视角的信息欠定度(非噪声)。
./SYNTHESIS_METHODOLOGY.md:52:- **机制(自然长出)**: 每张图建成分布(均值+方差), 航拍方差大/地面方差小; 包含损失让航拍分布⊆地面分布(非对称, 不是拉近两个点)。
./SYNTHESIS_METHODOLOGY.md:53:- **证重定义**: ① 航拍图特征方差是否系统性 > 地面(分布可视化); ② 非对称包含距离是否同时 beat 对称cosine 和 MaxSim; ③ 只在高视角差样本收益更大。
./SYNTHESIS_METHODOLOGY.md:55:- **kill-switch(零训练)**: frozen Swin(swin_fix256, 67.33)提特征, CARGO A↔G: 对称cosine vs 非对称包含距离(航拍当宽高斯/地面窄高斯, 用马氏或KL包含)。包含明显赢 → re-framing 有腿; 打平 → 死, 回头。
./SYNTHESIS_METHODOLOGY.md:58:- **观察**: avg>MaxSim 说明强行局部对齐(MaxSim)在极端跨视角**有害**。
./SYNTHESIS_METHODOLOGY.md:59:- **重定义**: "航拍和地面不共享可对齐的**局部**, 只共享**全局身份**+**视角特有**判别线索; 强行对齐(MaxSim)塌掉视角特有线索, 均匀平均(avg)稀释它。" → 学"视角中性但保判别"特征。
./SYNTHESIS_METHODOLOGY.md:61:- **kill-switch**: frozen Swin, 中性特征(去视角分量)+视角特有残差 分开重组 vs avg。
./SYNTHESIS_METHODOLOGY.md:66:- **kill-switch**: NCC(view→X) 因果强度, do-intervention 前后 per-view ID 可分性。
./SYNTHESIS_METHODOLOGY.md:68:### 候选 E(弱, 记录): 表示形态 + SMPL view-canonical
./SYNTHESIS_METHODOLOGY.md:69:- "Beyond geometry"(deep_3#6)已做 UVTexture canonical-viewpoint for aerial-ground(AG-ReID.v2)→ **直接撞车**, 降优先级。
./SYNTHESIS_METHODOLOGY.md:72:启 4-5 个 codex(--search)对候选 B/C/D 做: ① novelty/撞车检索(PDA/OT-ReID/CycleTrans/causal-ReID 边界); ② 哪个最强 + 为什么; ③ kill-switch 设计是否真能证伪。**绝不动手前先查 novelty + 必须有廉价 kill-switch(铁律)。**
./pivot/p_3.md:14:背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。
./pivot/p_3.md:16:**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
./pivot/p_3.md:18:- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
./pivot/p_3.md:21:**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。
./pivot/p_3.md:23:**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。
./pivot/p_3.md:26:角色=**残酷否决者**。不留情面判断: 整个 CARGO/aerial-ground empirical 方向, 对一篇 B 类**方法稿**(不是 benchmark/不是 backbone), 是不是已经是死胡同?证据: 遮挡5死角→转empirical→OVLI/OVP/MaxSim/containment 又全倒, method 始终不成形, 只有 avg 这个平凡 baseline 撑着。联网核查 aerial-ground ReID 2024-2026 现状(还有没有 method 空间, 还是已被 GSAlign/VDT/AG-VPReID 这代占满)。如果是死胡同, 明确说'该放弃换战场', 并指出团队最该回到哪类问题(基于它的真实资产: SMPL几何/Swin/遮挡历史)。如果不是死胡同, 指出唯一还值得赌的点。务实中文, 别和稀泥。
./pivot/p_3.md:65:../../exp109/design.md
./pivot/p_3.md:66:../../exp109/monitor.md
./pivot/p_3.md:71:../../exp149/design.md
./pivot/p_3.md:72:../../exp149/monitor.md
./pivot/p_3.md:73:../../exp149/claude_review.md
./pivot/p_3.md:74:../../exp148/design.md
./pivot/p_3.md:75:../../exp148/monitor.md
./pivot/p_3.md:76:../../exp148/claude_review.md
./pivot/p_3.md:92:-rw-r--r--@  1 abdslm  staff  15674 Jun 24 16:15 cvpb_containment_full.log
./pivot/p_3.md:93:-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
./pivot/p_3.md:226:drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp109
./pivot/p_3.md:264:drwxr-xr-x@   7 abdslm  staff     224 Jun 14 21:12 exp148
./pivot/p_3.md:265:drwxr-xr-x@   7 abdslm  staff     224 Jun 14 21:12 exp149
./pivot/p_3.md:460:drwxr-xr-x@   6 abdslm  staff     192 Jun 18 14:23 exp333_smpl3d
./pivot/p_3.md:511:我找到了上级实验文档和当前 `litreview2` 里的 containment kill-switch 材料。接下来只抽读与“是否换战场”有关的结论段，不展开全量文档。
./pivot/p_3.md:588:| 035b | PSG+GCN score*visibility weight | 60.4% | 71.6% | 84.8% | 87.9% | mAP+3.8%, R1+5.1% | ❌ vs 035a: -0.7% mAP, -2.2% R1。当前只说明 `score*visibility` 未带来收益，不能上升为整条 visibility 路线结论 |
./pivot/p_3.md:592:| 036 | PSG+GCN + Per-Keypoint Triplet Loss | 60.6% | 73.1% | 84.5% | 88.2% | mAP+4.0%, R1+6.6% | ❌ vs 035a: -0.5% mAP, -0.7% R1。该编号已偏离原 visibility 路线，实际属于 `exp035` 之后的 branch 内部探索 |
./pivot/p_3.md:593:| 037 | PSG+GCN + Learnable Keypoint Attention | 60.7% | 71.7% | 83.8% | 87.1% | mAP+4.1%, R1+5.2% | ❌ vs 035a: -0.4% mAP, -2.1% R1。该编号已偏离原 visibility 路线，LKA 未显示稳定正增益 |
./pivot/p_3.md:594:| 039a | PSG+GCN + CVK retrieval (`cvk_only`) | 59.3% | 72.9% | 84.1% | 87.1% | mAP+2.7%, R1+6.4% | 测试时诊断；vs 035a: -1.8% mAP, -0.9% R1。纯共同可见关键点距离不足以替代 `equal_concat` |
./pivot/p_3.md:595:| 039b | PSG+GCN + CVK retrieval (`cvk_hybrid`) | 61.9% | 73.2% | 85.2% | 88.5% | mAP+5.3%, R1+6.7% | 测试时诊断；vs 035a: +0.8% mAP, -0.6% R1。共同可见关键点更适合作为 global 的 pair-specific 补充 |
./pivot/p_3.md:680:| 148-eq | PSG+GCN+PCVT (Pose-Complementary View Training, eq) | ~59.3%* | ~71.3%* | — | — | ❌ 负面。*ep100 数据，训练中。早期加速（ep30: +2.4 mAP）但后期被基线追平并反超。3-view 训练的 1/3 主损失稀释导致后期收敛不足。训练集 95.8% 全可见使 complementary masking 缺乏信号 |
./pivot/p_3.md:682:| 151-eq | PSG+GCN+PVAT (Pose-Visibility Adversarial Training, eq) | 进行中 | — | — | — | 🟡 中性趋势。ep70: 59.0/72.0 vs exp030a 58.1/70.9 (+0.9/+1.1)。但 pvat_acc=0.83 不降——训练集 95.8% 可见，adversarial 无信号。预计最终中性 |
./pivot/p_3.md:683:| **maxsim** | **exp030a + MaxSim (ColBERT-style late interaction)** | **60.1%** | **74.4%** | **84.3%** | **87.5%** | **🟢 Test-time method。R1 74.4% 最高！但 mAP 低于 equal_concat (-1.0%)** |
./pivot/p_3.md:684:| **maxsim_hybrid 1:1** | **exp030a + MaxSim Hybrid (global+maxsim)** | **62.2%** | **73.8%** | **84.9%** | **88.2%** | **🟢🟢 超越 CVK hybrid (61.9/73.2)！mAP+1.1% vs eq_concat** |
./pivot/p_3.md:685:| **maxsim_hybrid 1:2** | **exp030a + MaxSim Hybrid (偏向 MaxSim)** | **62.2%** | **74.5%** | — | **88.6%** | **🟢🟢🟢 mAP+1.1, R1+0.8 vs eq_concat。ColBERT-style late interaction** |
./pivot/p_3.md:686:| **maxsim_paa 1:2** | **PAA (exp066) + MaxSim Hybrid** | **62.6%** | **75.2%** | **85.6%** | **89.0%** | **🟢🟢 vs PAA eq_concat(61.6/74.2): +1.0/+1.0** |
./pivot/p_3.md:687:| **maxsim_paa_roa 1:2** | **PAA+ROA (exp067) + MaxSim Hybrid** | **63.5%** | **75.4%** | **86.2%** | **88.9%** | **🟢🟢🟢🟢 vs PAA+ROA eq_concat(62.0/73.7): +1.5/+1.7。跨 checkpoint 稳定正向** |
./pivot/p_3.md:688:| 152b-eq | MaxSim Hard Triplet Training (tau=0.005, eq_concat) | 57.8% | 69.7% | — | 86.8% | ❌ vs exp030a-eq: **-3.3/-4.0**。MaxSim training 严重损害特征 |
./pivot/p_3.md:689:| 152b-ms | MaxSim Hard Triplet Training (maxsim_hybrid 1:2) | 59.0% | 71.0% | 83.8% | 87.2% | ❌ vs exp030a maxsim: **-3.2/-3.5**。即使 MaxSim test 也无法回补 |
./pivot/p_3.md:690:| 152-eq | MaxSim Soft Triplet Training (tau=0.05, eq_concat) | 57.8% | 70.3% | — | 87.4% | ❌ vs exp030a-eq: **-3.3/-3.4**。与 hard 版结果一致 |
./pivot/p_3.md:691:| 153-eq | MaxSim Additive w=0.25 (eq_concat) | 60.6% | 72.3% | — | 88.0% | 🟡 中性 vs exp030a-eq: **-0.5/-1.4**。不有害但无增益 |
./pivot/p_3.md:692:| 153-ms | MaxSim Additive w=0.25 (maxsim_hybrid 1:2) | 61.8% | 74.3% | 85.1% | 88.4% | 🟡 中性 vs exp030a maxsim: **-0.4/-0.2** |
./pivot/p_3.md:693:| 153b-eq | MaxSim Additive w=1.0 (eq_concat) | 57.6% | 70.0% | — | 87.1% | ❌ vs exp030a: **-3.5/-3.7**。w=1.0 崩了，与 replace 模式一致 |
./pivot/p_3.md:700:| **157-ms** | **PLBOA + MaxSim hybrid 1:2** | **64.1%** | **75.0%** | **86.4%** | **89.8%** | **🟢🟢🟢🟢 项目最高！vs baseline maxsim: +1.9/+0.5** |
./pivot/p_3.md:743:**上下文**: Phase 1 (33 个实验) 已充分探索了基于 ViTPose visibility 向量的所有方向。最佳训练端改进仅 +1.4%（GiLt+PCFC）。用户指示放弃 visibility 方向，转向纯 pose heatmap + mmpose 更鲁棒模型。
./pivot/p_3.md:977:这轮重审后，PRCV 主故事优先回到 `PSG`，而不是继续把 `LGPA-D + MaxSim + flip` 当主创新。
./pivot/p_3.md:984:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
./pivot/p_3.md:1050:   - `MaxSim / POT / flip`
./pivot/p_3.md:1097:| Backbone | Method | mAP (eq) | R1 (eq) | mAP (MaxSim) | R1 (MaxSim) |
./pivot/p_3.md:1126:4. **MaxSim Hybrid Matching**
./pivot/p_3.md:1143:   - 所有 visibility-dependent 训练方法失败 (VCSR, routing)
./pivot/p_3.md:1151:> 配合 PSG (backbone 内 pose 注入) 和 MaxSim (part-level late interaction matching)，
./pivot/p_3.md:1169:1. **MaxSim Behavior on Tiny**: `MaxSim` 的收益更依赖 per-keypoint consistency，而不是简单取决于 global 强弱。
./pivot/p_3.md:1202:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
./pivot/p_3.md:1204:   - `MaxSim/POT/flip` 主要是 test-time
./pivot/p_3.md:1239:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
./pivot/p_3.md:1247:- ViTPose visibility 向量不够可靠（AP 相关性仅 0.237）
./pivot/p_3.md:1248:- 中间层 visibility modulation 有害（破坏预训练空间结构），但这是 visibility 特有问题
./pivot/p_3.md:1251:- **关键结论**: 不要用 visibility 向量，用原始 pose 热图
./pivot/p_3.md:1345:可见部位特征 沿骨骼边传播到遮挡部位
./pivot/p_3.md:1409:本地旧文档显示了一条很长的“姿态/遮挡 ReID”探索史，且许多后续实验已经在反复验证“加结构/补全/metric/loss 很难形成新机制”。我会再读你这轮 CARGO empirical 的 kill-switch 和综合材料，判断是否只是单次失败还是路线失败。
./pivot/p_3.md:1411:/bin/zsh -lc "sed -n '1,240p' cvpb_containment_killswitch_design.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./pivot/p_3.md:1413:# 候选 B 零训练 kill-switch 设计（cvpb_containment_killswitch.py）
./pivot/p_3.md:1416:检验 re-frame："航拍-地面 ReID 不是对称匹配，而是物理定向的不确定性包含"。
./pivot/p_3.md:1423:3. 收益来自图像级非对称包含，不是混杂：8 个破坏对照全部掉分。
./pivot/p_3.md:1428:- 包含距离(A→G, query 航拍 a, gallery 地面 g)：D = KL(N_g‖N_a)，升序检索。
./pivot/p_3.md:1440:按航拍 query bbox 面积(area_of=h·w)分 4 桶(CARGO 无 A0/A1/A2 altitude 文件夹，故用面积)，报每桶 cosine vs 正确包含 mAP，预期最小面积桶包含收益最大。
./pivot/p_3.md:1452:true-pair 包含距离 << impostor。全过 = 隐藏变量证实；任一关键条不过 = B 降级。
./pivot/p_3.md:1491:2. **假设2 部分**: KL(g‖a) A→G=68.62 仅 +1.2 over cosine, 但 equal-var Maha(σ全常量)=67.94 同样 > cosine → "增益"来自 σ-free 距离形式, 不是包含。最佳对称分布距离(sym-KL/JS/Bhatt)全 << cosine。
./pivot/p_3.md:1493:- **方向不对称是检索 artifact**: 真正起作用的是"均值项除以 query 端方差"(A→G query=航拍→好; G→A query=地面→好), 除以 gallery 端方差必崩(高方差 gallery 永远排前)。G→A CORRECT(除 gallery-aerial σ)崩到 17.37 = 铁证。这不是物理包含, 是分母选谁的归一化效应。
./pivot/p_3.md:1497:/bin/zsh -lc 'tail -n 220 cvpb_containment_full.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./pivot/p_3.md:1527:| 10 | **非对称包含** | PDA: 文本分布⊇图像分布; 方差=语义范围(非噪声) |
./pivot/p_3.md:1529:| 12 | **表示形态错了** | 3D Gait: 稀疏SMPL参数没法和稠密appearance融→蒸成稠密时空场; "换辅助模态"消融证 dense>skeleton>SMPL参 |
./pivot/p_3.md:1543:## 三、应用到我们自己(候选 re-framing, 每个带廉价 kill-switch)
./pivot/p_3.md:1545:我们独有资产: CARGO/AG-ReID.v2 极端跨视角(航拍↔地面 90°, 航拍低清俯视); **观察: avg-pool 52.37 > token-MaxSim 45.19(差7分)**; SMPL 几何基建; Swin 67.33(backbone)。
./pivot/p_3.md:1547:> ⚠️ 用户已打掉"MaxSim<avg→局部不可靠→用avg"(推到底=零贡献)。**新候选的硬门槛: 机制必须 beat avg, 不能退化成 avg。** aerial-ground 几何/可见性是红海(避开)。
./pivot/p_3.md:1549:### 候选 B(主推): 非对称包含 — 把对称匹配换成"航拍证据⊆地面证据"
./pivot/p_3.md:1550:- **观察**: 航拍低清俯视 = 信息**欠定**(看不清脸/纹理); 地面高清正面 = 信息**确定**。但所有 cross-view ReID 用**对称** cosine/MaxSim 匹配, 默认两视角信息对等。
./pivot/p_3.md:1551:- **重定义(move 10+22)**: "大家以为跨视角是对称对齐问题, 其实航拍↔地面是**非对称信息包含**: 航拍身份证据是一个**范围(宽分布)**, 应被地面的**窄分布包含**。" 方差=该视角的信息欠定度(非噪声)。
./pivot/p_3.md:1552:- **机制(自然长出)**: 每张图建成分布(均值+方差), 航拍方差大/地面方差小; 包含损失让航拍分布⊆地面分布(非对称, 不是拉近两个点)。
./pivot/p_3.md:1553:- **证重定义**: ① 航拍图特征方差是否系统性 > 地面(分布可视化); ② 非对称包含距离是否同时 beat 对称cosine 和 MaxSim; ③ 只在高视角差样本收益更大。
./pivot/p_3.md:1555:- **kill-switch(零训练)**: frozen Swin(swin_fix256, 67.33)提特征, CARGO A↔G: 对称cosine vs 非对称包含距离(航拍当宽高斯/地面窄高斯, 用马氏或KL包含)。包含明显赢 → re-framing 有腿; 打平 → 死, 回头。
./pivot/p_3.md:1558:- **观察**: avg>MaxSim 说明强行局部对齐(MaxSim)在极端跨视角**有害**。
./pivot/p_3.md:1559:- **重定义**: "航拍和地面不共享可对齐的**局部**, 只共享**全局身份**+**视角特有**判别线索; 强行对齐(MaxSim)塌掉视角特有线索, 均匀平均(avg)稀释它。" → 学"视角中性但保判别"特征。
./pivot/p_3.md:1561:- **kill-switch**: frozen Swin, 中性特征(去视角分量)+视角特有残差 分开重组 vs avg。
./pivot/p_3.md:1566:- **kill-switch**: NCC(view→X) 因果强度, do-intervention 前后 per-view ID 可分性。
./pivot/p_3.md:1568:### 候选 E(弱, 记录): 表示形态 + SMPL view-canonical
./pivot/p_3.md:1569:- "Beyond geometry"(deep_3#6)已做 UVTexture canonical-viewpoint for aerial-ground(AG-ReID.v2)→ **直接撞车**, 降优先级。
./pivot/p_3.md:1572:启 4-5 个 codex(--search)对候选 B/C/D 做: ① novelty/撞车检索(PDA/OT-ReID/CycleTrans/causal-ReID 边界); ② 哪个最强 + 为什么; ③ kill-switch 设计是否真能证伪。**绝不动手前先查 novelty + 必须有廉价 kill-switch(铁律)。**
./pivot/p_3.md:1782:  (expectation: SMALLEST area bucket -> largest containment gain)
./pivot/p_3.md:1794:[done] kill-switch complete. See B_CONTAINMENT_DESIGN.md sec.4 pass criteria.
./pivot/p_3.md:1799:# 候选 B 定稿: 航拍-地面 ReID 的"物理定向不确定性包含" + 零训练 kill-switch
./pivot/p_3.md:1806:| v_1 | 蓝队杀手 | B **存活但裸想法被打惨**: 无直接撞车, 但机制三件套(高斯/方差不确定性/KL-偏序包含)全有强先例(Word2Gauss/Order-Emb/HIB/**PFE**/PCME/**Pr-VIPE**/LPP-ReID)。只能当"任务重定义"卖, 不能当"概率/KL 机制"卖。 |
./pivot/p_3.md:1807:| v_2 | 红队辩护 | **信心 7/10**(过双 kill-switch→8)。B > C/D(C 退化成"别做局部匹配", D 难证 view-confounder)。headline 必须是"揭示物理定向信息不对称", 不是"我们提出 Gaussian/KL"。**训练版必须 beat avg 52.37**(不只 MaxSim)。 |
./pivot/p_3.md:1808:| v_3 | 独立裁判 | 给了更硬的 re-frame: **观测受限身份可恢复性**(avg>MaxSim 真意=低清航拍局部 token 是欠定噪声, MaxSim 捞假局部匹配)。C **撞 AGPReID 近作 ViSA**。B 排第一但方向要修。 |
./pivot/p_3.md:1809:| v_4 | kill-switch 批判 | 裸 kill-switch 不够, 给了 **8 个破坏性对照**。建议**先做加固版 B**。 |
./pivot/p_3.md:1811:**三方独立确认的硬修正: 包含方向写反了。** 物理上"航拍证据 ⊂ 地面外观集合"; 概率候选空间上 = **地面窄分布落入航拍宽分布(G ⊂ A)**, 打分 `-KL(N_G ‖ N_A)`。原 SYNTHESIS 写的"航拍⊆地面"是反的(=惩罚不确定性, 和叙事冲突)。
./pivot/p_3.md:1815:> **Cross-view (aerial-ground) ReID 不是对称对齐问题, 而是物理定向的不确定性包含。**
./pivot/p_3.md:1816:> 航拍是受像素预算限制的欠定投影: 其身份兼容的地面外观构成一个**更宽的候选分布**; 正确的地面证据应**落在这个航拍不确定性包络内**。对称 cosine / late-interaction(MaxSim) 用了错误的匹配假设——把欠定的航拍局部 token 当成可独立匹配的身份证据, 于是从 gallery 捞偶然高相似的**假局部匹配**(这就是 avg 52.37 > MaxSim 45.19 的真因, 不是"该用 avg")。
./pivot/p_3.md:1818:**隐藏变量(可测、可证伪)**: 视角成像导致的**信息欠定度不对称** σ_aerial ≫ σ_ground, 且 σ 由物理量(bbox 面积/SMPL 投影身体像素/俯视比/分辨率)决定, 不是难度代理。
./pivot/p_3.md:1825:- vs **AGPReID 红海**(VDT 解耦/GSAlign TPS+可见性/SeCap prompt/DTST token选择/**ViSA 视角特有线索**): 避开几何/可见性/局部选择; **SMPL 只当诊断物理欠定度, 不当主模块**。
./pivot/p_3.md:1826:- vs **cross-resolution ReID**(PS-HRNet/RFD 恢复/不变特征): 不幻想补不可见细节, 而是**显式表达候选身份范围**。
./pivot/p_3.md:1829:## 四、★零训练 kill-switch 协议（加固版, v_4 主导 + v_2/v_3 补）
./pivot/p_3.md:1831:冻结同一 Swin(swin_fix256, 67.33), 不训练, CARGO A→G。三条核心假设, 任一不成立 B 降级:
./pivot/p_3.md:1835:- 诊断: trace(σ_A) 显著 > trace(σ_G); σ 与 bbox面积↓/SMPL投影像素↓/分辨率↓ 相关。
./pivot/p_3.md:1841:**假设3 — 收益来自图像级非对称包含, 不是混杂**: 8 个破坏性对照全部必须掉分:
./pivot/p_3.md:1843:2. 对称化破坏: sym-KL/JS 一样好→收益是 distribution metric 非包含。
./pivot/p_3.md:1849:8. 收益集中度: 按 SMPL投影像素/bbox面积/分辨率分 4 桶, 最小航拍桶包含收益最大, 高清/同视角弱→全桶平均涨=普通 metric trick。
./pivot/p_3.md:1853:**通过标准**: -KL(G‖A) > cosine 且 > 最佳对称; 正向 >> 反向; 对照3 的 1/3/4/5/6/7 全明显掉; σ_A≫σ_G 且合成退化单调升 σ; true-pair 包含距离 << impostor 且差距在破坏对照里消失。
./pivot/p_3.md:1856:## 五、训练版门槛（kill-switch 过后才做, v_2）
./pivot/p_3.md:1857:方法朴素即可: mean+variance head + directional containment loss + 物理欠定正则 + 包含检索分。
./pivot/p_3.md:1858:**判据: 单 seed ≥ +1.0 mAP over avg 52.37, 低清 A→G 分桶 +2~3 mAP。** 体量来自诊断证据+破坏对照+两数据集(CARGO+AG-ReID.v2), 不是模块复杂度。
./pivot/p_3.md:1861:> We reveal a **physically directed information asymmetry** in aerial-ground person ReID, and show that symmetric alignment / late-interaction is the wrong matching assumption: the low-altitude aerial observation is an under-determined projection whose identity evidence must be matched by **directional containment**, not symmetric similarity.
./pivot/p_3.md:1864:写零训练 kill-switch 脚本(复用 error_analysis_geom.py 基建: frozen Swin 提 μ + σ, CARGO A↔G, 8 破坏对照 + SMPL 分桶)。GPU: lab-3090(CARGO + swin_fix256 ckpt 在)。先双审? —— 这是零训练诊断脚本不是训练, 但仍走 codex 审一遍脚本正确性再跑。
./analyses/deep_13.txt:4:1. **触发观察**：VI-ReID 里大家都在学 modality-shared feature，但这样会丢掉一部分有判别力的 modality-specific 信息。已有补偿方法走图像生成，RGB 和 IR 互转太难，尤其 IR 生成可见光会带来颜色不一致等噪声，所以补偿线反而不如 shared feature 线。
./analyses/deep_13.txt:42:4. **证据闭环**：Fig. 3 先证明旧方法早期 gradient consistency 低。update direction 消融比较 current gradient、GAG 和 weighted gradient，weighted gradient 最好。inner iteration 数 m 和噪声边界 beta 的实验展示存在合理区间，不是随便加算力。黑盒、白盒、ensemble、防御模型、跨模态 ReID 都验证有效，检索可视化显示 adversarial query 会让 top-10 全错。
./pivot/p_2.md:14:背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。
./pivot/p_2.md:16:**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
./pivot/p_2.md:18:- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
./pivot/p_2.md:21:**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。
./pivot/p_2.md:23:**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。
./pivot/p_2.md:26:角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。
./pivot/p_2.md:28:我会先读本地实验脉络，确认哪些路线已经被文档明确判负；同时联网做撞车核查，避免把已有的几何对齐/可见性路线换个名字再提。
./analyses/deep_18.txt:7:3. **机制怎么长出来**：机制基本是顺着重定义推出来的。既然 ReID 是 query 查 support，就把每个身份的可见光和红外样本循环拆成 query/support，让每张图都轮流参与。既然单 batch 不够全局，就用 memory bank 存历史特征，把历史样本当扩展 support。SMR 再约束 V2I 和 I2V 的预测一致性。
./analyses/deep_18.txt:21:1. **触发观察**：现有 VI-ReID 默认是可见光 query 查红外 gallery，或反过来。但真实系统里 query 和 gallery 往往都是混合模态。旧方法一到混合模态场景就出现 `modality confusion`，同模态不同身份可能比跨模态同身份更近。
./analyses/deep_18.txt:29:1. **触发观察**：正式版进一步把观察说得更工程化，单模态 ReID 和传统 VI-ReID 都没有覆盖“query 集和 gallery 集同时混合可见光与红外”的情况。作者强调 Rank-1 有时会因为同模态易匹配而变好，但 mAP 和 mINP 下降，说明整体检索质量变差。
./pivot/p_1.md:14:背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。
./pivot/p_1.md:16:**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
./pivot/p_1.md:18:- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
./pivot/p_1.md:21:**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。
./pivot/p_1.md:23:**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。
./pivot/p_1.md:26:角色=**救援者**。给定新发现 σ_aerial<σ_ground(航拍更平滑/少细节, 地面细节丰富)+ avg>MaxSim 这两个硬事实, 用联网搜索找一个**还活着的 B 类 re-frame**, 必须同时满足: (a)符合'航拍平滑低细节'而非'航拍噪声/欠定'; (b)不 reduce to avg(机制不能退化成平均池化); (c)有廉价零训练 kill-switch; (d)不撞 GSAlign/VDT/SeCap/ViSA/cross-resolution 红海。逐个候选查 novelty。如果找不到符合的, 诚实说'救不动'。务实中文, 给候选+kill-switch+撞车核查。
./pivot/p_1.md:28:我会先核对近年航拍-地面 ReID、跨分辨率 ReID、token/patch 聚合与池化相关方法，重点看是否已经把“航拍平滑低细节”当作核心问题来建模。随后按你的四个约束筛候选，并给每个候选配一个零训练 kill-switch。
./pivot/p_1.md:30:web search: aerial ground person re-identification AG-ReID CARGO GSAlign VDT SeCap ViSA paper
./pivot/p_1.md:38:初步检索确认几个红海边界：VDT/ViSA/SeCap/GSAlign都把核心矛盾写成视角、语义/局部对齐、几何错位或 view-aware token/expert；频域思路在 VI-ReID 和跨分辨率 ReID 里很多，但在 AG-ReID 里还没看到把“航拍更平滑、地面细节过剩”作为主问题来立论。下一步我会专门查“低频/纹理/细节过剩/shortcut/augmentation”一类路线，避免把候选变成普通跨分辨率。
./analyses/deep_11.txt:7:2. 重定义动作：他们把问题从“怎么匹配可见光和红外簇”改成“跨模态匹配关系本身可以反过来诊断和修正伪标签”。关键词是 modality-shared pseudo labels、cluster relationship-based label refinement、weighted modality-shared memory、hard samples。这里的重点不是又做一个匹配算法，而是把匹配后的 many-to-one 关系解释成伪标签错误的信号。
./analyses/deep_11.txt:29:1. 触发观察：USL-VI-ReID 里，大家主要盯着 visible-infrared modality gap，但作者观察到 clustering 阶段还有 camera discrepancy。可见光图像受相机、光照、视角影响更大，导致同一身份被过度拆裂。已有 graph matching 或 optimal transport 如果建立在拆裂簇上，反而会放大 noisy labels。
./analyses/deep_11.txt:47:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./analyses/deep_11.txt:65:1. 触发观察：VI-ReID 里很多方法都试图补齐或消除可见光和红外的信息差，比如生成、对齐、模态不变特征。作者反过来指出，完全桥接信息差很难，而且即使桥接，也不一定能稳定抓到身份判别线索。真正挑战是模型面对不同模态、不同实例、不同 query 时都要能灵活适配。
./analyses/deep_22.txt:7:2. **重定义动作**：它把问题从“图文相似度对齐”改写成“文本分布应当包含图像分布”。关键词是 probabilistic containment、semantic scope、partial-order relationship、text as probabilistic envelope。厉害的地方是把方差从“噪声余量”重新定义成“语义覆盖范围”。
./analyses/deep_22.txt:9:3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。
./analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./analyses/deep_22.txt:13:5. **reviewer 为什么买账**：这篇卖的是视角，不只是机制。它不是又加一个对齐头，而是把 TBPR 的匹配关系从点对点改成非对称包含。novelty 来自“方差等于语义范围”这个解释，以及用闭式包含损失把这个解释落成可训练目标。
./analyses/deep_22.txt:53:1. **触发观察**：VI-ReID 的可见光和红外图像有大模态差异，视觉特征很难直接对齐。已有 CLIP prompt 方法说明语义空间有帮助，但可学习 prompt 语义不透明、太粗、还要两阶段训练。LLaVA 能生成显式细粒度描述，但跨模态描述会有冲突，比如同一衣服在可见光里是蓝色，在红外里像灰色；同一身份不同视角的描述又互补。
./analyses/deep_22.txt:57:3. **机制怎么长出来**：ESFA 用 LLaVA 给可见光和红外图像生成描述，在各自模态内做图文对齐。CSIA 不直接强拉图像和异模态文本，而是约束“模态内图文相似度”和“跨模态文本相似度”一致，绕开颜色冲突。CVSC 用同身份其他视角文本补偿当前文本，把一对一图文匹配扩成一对多语义对齐。机制和重定义绑定非常紧。
./reviews/deep_27.md:487:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
./reviews/deep_27.md:504:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
./reviews/deep_27.md:757:检查脚本 `.codex/hooks/check_design.sh` 会在包含 `train.py` 的命令执行前检查：设计文档在不在，`review-claude.md` 是不是通过且至少三十行，`review-codex.md` 是不是通过。如果 `design.md` 里写了“需要训练前审查：否”，就只检查设计文档。
./reviews/deep_27.md:826:绝对不要用宽泛匹配的 `pkill -f codex`，那会误杀 Mac 上的 Codex.app。要杀进程就精确匹配。

exec
/bin/zsh -lc "sed -n '1,220p' pivot/p_3.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef8b9-f5cf-7783-99e9-1ce572c5dcfb
--------
user
背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。

**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
- ①σ_aerial < σ_ground(航拍q156.96/g167.47 < 地面q171.64/g172.81双侧; 合成退化σ反而降115.41<118.93)。'航拍更欠定=宽分布'前提**错的**——航拍低清=少高频细节=更平滑=低TTA方差。
- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
- ③image-level σ无用(C3 view-mean 69.07不降反升/C4同视角置换67.47/C5 hardness 66.63 都≈correct不掉)。

**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。

**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。

== 你的任务 ==
角色=**残酷否决者**。不留情面判断: 整个 CARGO/aerial-ground empirical 方向, 对一篇 B 类**方法稿**(不是 benchmark/不是 backbone), 是不是已经是死胡同?证据: 遮挡5死角→转empirical→OVLI/OVP/MaxSim/containment 又全倒, method 始终不成形, 只有 avg 这个平凡 baseline 撑着。联网核查 aerial-ground ReID 2024-2026 现状(还有没有 method 空间, 还是已被 GSAlign/VDT/AG-VPReID 这代占满)。如果是死胡同, 明确说'该放弃换战场', 并指出团队最该回到哪类问题(基于它的真实资产: SMPL几何/Swin/遮挡历史)。如果不是死胡同, 指出唯一还值得赌的点。务实中文, 别和稀泥。
codex
我会先核查近两三年的 aerial-ground / cross-view person ReID 方法线，重点看是否还有未被占掉的方法空间，而不是只按你给的内部结果下判断。
web search: 
web search: 2024 2025 aerial-ground person re-identification GSAlign VDT SeCap AG-VPReID ViSA
web search: 
web search: GSAlign aerial ground person re-identification paper
web search: 
web search: "ViSA" "Aerial-Ground" "Person Re-identification"
codex
我还会快速读本地最新结果/决策文档，避免只依据你这段摘要作判断。现在环境是只读，所以我不会改文档或提交。
exec
/bin/zsh -lc "pwd && rg --files experiments | rg '("'^experiments/(results|decisions|innovation_brainstorm)'"\\.md"'$|''^experiments/paper_materials/story'"\\.md"'$|''^experiments/exp(148|149|109|333)/(design|monitor|claude_review)'"\\.md"'$)'"'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
rg: experiments: IO error for operation on experiments: No such file or directory (os error 2)

exec
/bin/zsh -lc 'git status --short' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-CCanqUKT' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-fNKRTZA9' (errno=Operation not permitted)
 M ../../decisions.md
 M ../../../model/backbones/swin_transformer.py
?? ../../afd_reid/
?? ../
?? ../../cbcl_t2i/
?? ../../cross_view_cargo/
?? ../../library_167_synthesis.md
?? ../../publishable_angles_synthesis.md
?? ../../smpl_anchor_vireid/

codex
本地路径不是仓库根，`experiments/` 在当前目录下不存在；我会按当前目录相对位置去找上级实验文档。
exec
/bin/zsh -lc "rg --files ../.. | rg '("'^|/)(results|decisions|innovation_brainstorm|library_167_synthesis|publishable_angles_synthesis)'"\\.md"'$|paper_materials/story'"\\.md"'$|exp(109|148|149|333)/(design|monitor|claude_review)'"\\.md"'$'"'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
../../exp109/design.md
../../exp109/monitor.md
../../decisions.md
../../prcv_2026_psg/decisions.md
../../innovation_brainstorm.md
../../library_167_synthesis.md
../../exp149/design.md
../../exp149/monitor.md
../../exp149/claude_review.md
../../exp148/design.md
../../exp148/monitor.md
../../exp148/claude_review.md
../../results.md
../../publishable_angles_synthesis.md
../../paper_materials/story.md

exec
/bin/zsh -lc 'ls -la && ls -la ../..' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
total 152
drwxr-xr-x@ 15 abdslm  staff    480 Jun 24 16:23 .
drwxr-xr-x@ 65 abdslm  staff   2080 Jun 24 15:04 ..
-rw-r--r--@  1 abdslm  staff   6604 Jun 24 14:34 B_CONTAINMENT_DESIGN.md
-rw-r--r--@  1 abdslm  staff   7368 Jun 24 14:20 SYNTHESIS_METHODOLOGY.md
-rw-r--r--@  1 abdslm  staff  16038 Jun 24 13:58 all_papers.txt
drwxr-xr-x@ 31 abdslm  staff    992 Jun 24 14:13 analyses
drwxr-xr-x@ 30 abdslm  staff    960 Jun 24 13:58 batches
-rw-r--r--@  1 abdslm  staff  15674 Jun 24 16:15 cvpb_containment_full.log
-rw-r--r--@  1 abdslm  staff   5992 Jun 24 16:15 cvpb_containment_killswitch_design.md
-rw-r--r--@  1 abdslm  staff   2385 Jun 24 13:58 launch_30deep.sh
drwxr-xr-x@  5 abdslm  staff    160 Jun 24 16:23 pivot
drwxr-xr-x@ 30 abdslm  staff    960 Jun 24 13:59 reviews
-rw-r--r--@  1 abdslm  staff   4533 Jun 24 16:22 strategic_pivot.sh
drwxr-xr-x@  7 abdslm  staff    224 Jun 24 14:29 validate
-rw-r--r--@  1 abdslm  staff   4945 Jun 24 14:22 validate_candidates.sh
total 1648
drwxr-xr-x@ 408 abdslm  staff   13056 Jun 22 15:29 .
drwxr-xr-x@  44 abdslm  staff    1408 Jun 19 01:29 ..
-rw-r--r--@   1 abdslm  staff   26628 Jun 17 02:37 .DS_Store
-rw-r--r--@   1 abdslm  staff       0 Jun 14 21:12 .gitkeep
-rw-r--r--@   1 abdslm  staff   16571 Jun 17 18:42 MORNING_REPORT.md
-rw-r--r--@   1 abdslm  staff    2130 Jun 20 09:29 POSE_INTO_CLIP_summary.md
drwxr-xr-x@  15 abdslm  staff     480 Jun 24 01:53 afd_reid
drwxr-xr-x@  65 abdslm  staff    2080 Jun 24 15:04 cargo_cvpb
drwxr-xr-x@   5 abdslm  staff     160 Jun 22 15:00 cbcl_t2i
drwxr-xr-x@   5 abdslm  staff     160 Jun 19 07:26 clip_reid_compare
drwxr-xr-x@   5 abdslm  staff     160 Jun 22 14:03 cross_view_cargo
-rw-r--r--@   1 abdslm  staff  271436 Jun 23 14:47 decisions.md
drwxr-xr-x@   3 abdslm  staff      96 Jun 14 21:12 exp000
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp000b
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp001
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp002
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp003
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp004
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp005
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp006
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp007
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp007a
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp007b
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp008
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp009
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp010
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp011
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp012
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp013
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp014
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp015
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp016
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp017
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp018
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp019
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp020
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp021
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp022
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp023
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp024
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp025
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp026
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp027
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp028
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp029
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp030
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp030a
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp030b
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp031
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp032
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp033
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp034
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp035
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp036
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp037
drwxr-xr-x@   3 abdslm  staff      96 Jun 14 21:12 exp038
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp039
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp040
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp041
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp042
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp043
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp044
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp045
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp046
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp047
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp048
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp049
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp050
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp051
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp052
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp053
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp054
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp055
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp056
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp057
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp058
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp059
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp060
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp061
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp062
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp063
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp064
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp065
drwxr-xr-x@   6 abdslm  staff     192 Jun 14 21:12 exp066
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp067
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp068
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp069
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp070
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp071
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp072
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp073
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp074
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp075
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp076
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp077
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp078
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp079
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp080
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp081
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp083
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp084
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp085
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp086
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp087
drwxr-xr-x@   3 abdslm  staff      96 Jun 14 21:12 exp088
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp089
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp090
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp091
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp092
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp093
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp094
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp095
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp096
drwxr-xr-x@   4 abdslm  staff     128 Jun 14 21:12 exp097
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp098
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp099
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp100
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp101
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp102
drwxr-xr-x@   5 abdslm  staff     160 Jun 14 21:12 exp103

web search: 
web search: "APTM" "Text-Image Person" "arXiv"
web search: 
web search: "CADA" "text-based person retrieval" "RSTPReid"
web search: 
web search: "CADA" "person re-identification" "text"
web search: 
web search: "RDE" "Noisy Correspondence" "Text-to-Image Person Re-identification"
web search: 
web search: "IRRA" "Text-to-Image Person Retrieval" "RSTPReid"
web search: 
web search: https://arxiv.org/abs/2303.12501
web search: 
web search: "Probabilistic Distribution Alignment" "Text-Based Person Retrieval"
web search: 
web search: "Probabilistic Distribution Alignment for Text-Based Person Retrieval"
web search: 
web search: "A training-free framework for text-to-image person re-identification via query-prototype matching"
web search: 
web search: RSTPReid dataset text-to-image person retrieval real scenario statistics images identities descriptions
web search: 
web search: "False Negatives Consensus Suppression" "Text-to-Image Person Re-identification"
web search: 
web search: "False Negatives Consensus Suppression" "person"
web search: 
web search: "UP-Person" "Text-Based Person Retrieval"
web search: 
web search: "Learning Visual-Semantic Embedding for Generalizable Person Re-Identification"
web search: 
web search: "Text-to-image person retrieval" "MLLM" "RSTPReid"
web search: 
web search: "Semantic Polymorphism and Commonality Learning" "Text-Based Person Retrieval"
codex
**Verdict：最值得押的一条是 RSTPReid 上的“ReID 视觉流形锚定文本检索”，信心 5/10。**  
不是做新 alignment 模块，而是重定义：**text-to-image ReID 不是把图像拖进语言空间，而是把欠完整文本定位到一个强视觉 ReID 身份流形上。** 这正好用你们的强 Swin/SOLIDER 视觉 ckpt；但只能先跑 kill-switch，过不了就别开训练，转 Hubness analysis short。

**触发观察**  
在 RSTPReid 上找这种失败：文本描述很泛，例如黑衣、深色裤、背包，标准 text-image 排名把一堆语义相似人排前；但目标图像在 frozen SOLIDER image-image 空间里其实有清晰身份邻域。也就是：失败不是“局部图文没对齐”，而是**文本查询没有落到已有视觉身份流形的正确区域**。

**机制**  
方法可叫 `RMA-TIReID`：ReID-Manifold Anchored Text-to-Image ReID。

- 冻结或强约束 SOLIDER/Swin image encoder，保住 image ReID manifold。
- 文本 encoder 不直接和图像一起自由漂移，而是学习预测/匹配 frozen visual ReID feature、ID prototype 或 teacher similarity distribution。
- 加一个 manifold-preserving loss：训练后的 image embedding 必须保持 frozen SOLIDER 的 gallery-gallery / same-ID topology。
- 文本侧用 attribute/token 组合去生成“视觉流形查询”，而不是再做 CADA/IRRA 式 patch-word 交互。
- 测试仍是单 text embedding 查 gallery image embedding，不用 MLLM、不用 re-ranking。

**廉价 kill-switch**  
不训练 backbone，先做两版：

1. 提取 RSTPReid 全部 image 的 frozen SOLIDER feature。
2. 用 train captions 建 phrase/token → visual direction/prototype 表：例如颜色、上衣、裤子、包、性别、长短袖等 token 的平均视觉 feature residual。
3. test text query = IDF 加权 token prototype sum，直接查 gallery SOLIDER features。
4. 控制组：随机 token-image 配对、color-only、去 color、caption shuffle、feature shuffle；如果有本地 CLIP/BERT cache，再加 CLIP zero-shot/linear probe。

通过门槛：这个零训练/轻训练 text→SOLIDER 查询必须明显强于 color-only 和 shuffle，并且错误主要发生在高文本歧义 query。若它只会认颜色，或接近随机，方向直接死。

**撞车核查**  
红线很明确：

- IRRA 已做 implicit local relation + global distribution matching，别再做普通图文局部对齐。([arxiv.org](https://arxiv.org/abs/2303.12501))
- CADA 已做 text-token↔image-patch / image-region↔text-attribute 双向 association。([arxiv.org](https://arxiv.org/abs/2312.01745?utm_source=openai))
- RDE 已覆盖 noisy correspondence / false-correlated image-text pair。([arxiv.org](https://arxiv.org/abs/2308.09911?utm_source=openai))
- APTM/UniPT/UP-Person 都走大规模预训练或 CLIP/PETL 路线；无下载约束下别硬碰。([arxiv.org](https://arxiv.org/abs/2306.02898?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2309.01420?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2504.10084?utm_source=openai))
- CAIBC 已讲 color over-reliance；如果你的结果只是“别只看颜色”，也撞。([arxiv.org](https://arxiv.org/abs/2209.05773?utm_source=openai))
- DiCo 已做 slot/concept disentanglement；别把贡献写成属性槽分解。([arxiv.org](https://arxiv.org/abs/2601.10053?utm_source=openai))
- ICL 已用 MLLM test-time query refinement，这条不符合 on-disk/no-download。([arxiv.org](https://arxiv.org/abs/2506.11036?utm_source=openai))

所以最终判断：**text-image 没卷死，但“普通图文方法”全死。唯一有团队资产差异化的是 SOLIDER 视觉身份流形当 teacher/anchor。** 先跑 kill-switch；失败就别再换壳，Hubness 写 analysis short 收尾更干净。
tokens used
199,801
**Verdict：最值得押的一条是 RSTPReid 上的“ReID 视觉流形锚定文本检索”，信心 5/10。**  
不是做新 alignment 模块，而是重定义：**text-to-image ReID 不是把图像拖进语言空间，而是把欠完整文本定位到一个强视觉 ReID 身份流形上。** 这正好用你们的强 Swin/SOLIDER 视觉 ckpt；但只能先跑 kill-switch，过不了就别开训练，转 Hubness analysis short。

**触发观察**  
在 RSTPReid 上找这种失败：文本描述很泛，例如黑衣、深色裤、背包，标准 text-image 排名把一堆语义相似人排前；但目标图像在 frozen SOLIDER image-image 空间里其实有清晰身份邻域。也就是：失败不是“局部图文没对齐”，而是**文本查询没有落到已有视觉身份流形的正确区域**。

**机制**  
方法可叫 `RMA-TIReID`：ReID-Manifold Anchored Text-to-Image ReID。

- 冻结或强约束 SOLIDER/Swin image encoder，保住 image ReID manifold。
- 文本 encoder 不直接和图像一起自由漂移，而是学习预测/匹配 frozen visual ReID feature、ID prototype 或 teacher similarity distribution。
- 加一个 manifold-preserving loss：训练后的 image embedding 必须保持 frozen SOLIDER 的 gallery-gallery / same-ID topology。
- 文本侧用 attribute/token 组合去生成“视觉流形查询”，而不是再做 CADA/IRRA 式 patch-word 交互。
- 测试仍是单 text embedding 查 gallery image embedding，不用 MLLM、不用 re-ranking。

**廉价 kill-switch**  
不训练 backbone，先做两版：

1. 提取 RSTPReid 全部 image 的 frozen SOLIDER feature。
2. 用 train captions 建 phrase/token → visual direction/prototype 表：例如颜色、上衣、裤子、包、性别、长短袖等 token 的平均视觉 feature residual。
3. test text query = IDF 加权 token prototype sum，直接查 gallery SOLIDER features。
4. 控制组：随机 token-image 配对、color-only、去 color、caption shuffle、feature shuffle；如果有本地 CLIP/BERT cache，再加 CLIP zero-shot/linear probe。

通过门槛：这个零训练/轻训练 text→SOLIDER 查询必须明显强于 color-only 和 shuffle，并且错误主要发生在高文本歧义 query。若它只会认颜色，或接近随机，方向直接死。

**撞车核查**  
红线很明确：

- IRRA 已做 implicit local relation + global distribution matching，别再做普通图文局部对齐。([arxiv.org](https://arxiv.org/abs/2303.12501))
- CADA 已做 text-token↔image-patch / image-region↔text-attribute 双向 association。([arxiv.org](https://arxiv.org/abs/2312.01745?utm_source=openai))
- RDE 已覆盖 noisy correspondence / false-correlated image-text pair。([arxiv.org](https://arxiv.org/abs/2308.09911?utm_source=openai))
- APTM/UniPT/UP-Person 都走大规模预训练或 CLIP/PETL 路线；无下载约束下别硬碰。([arxiv.org](https://arxiv.org/abs/2306.02898?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2309.01420?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2504.10084?utm_source=openai))
- CAIBC 已讲 color over-reliance；如果你的结果只是“别只看颜色”，也撞。([arxiv.org](https://arxiv.org/abs/2209.05773?utm_source=openai))
- DiCo 已做 slot/concept disentanglement；别把贡献写成属性槽分解。([arxiv.org](https://arxiv.org/abs/2601.10053?utm_source=openai))
- ICL 已用 MLLM test-time query refinement，这条不符合 on-disk/no-download。([arxiv.org](https://arxiv.org/abs/2506.11036?utm_source=openai))

所以最终判断：**text-image 没卷死，但“普通图文方法”全死。唯一有团队资产差异化的是 SOLIDER 视觉身份流形当 teacher/anchor。** 先跑 kill-switch；失败就别再换壳，Hubness 写 analysis short 收尾更干净。
