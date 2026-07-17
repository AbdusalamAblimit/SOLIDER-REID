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
session id: 019ef973-0b8e-7df2-9ba2-940180e6b5d3
--------
user
一个 ReID 团队连续 5 个 frozen-image/on-disk 方向被零训练 kill-switch 干净否掉(B航拍包含=前提错 / GOPL SMPL=occlusion-count代理 / Gallery Hubness=被k-reciprocal占 / OSAC谱坍缩=前提倒置训练反而升rank / RMA-TIReID text-image视觉流形锚定=边缘token-prototype≈color-only)。**frozen-feature + 现有 image/text on-disk 数据的 B 类方法空间穷尽。** 唯一真发现=gallery hubness 诊断(M(q)解释AP误差rho+0.60, analysis级非方法)。所有前序 codex 收敛: 唯一未测的 method-space 是**视频**(残酷终判推 AG-VPReID tracklet 时间证据 routing)。

但: 团队**无视频数据**(要下载 MARS ~4GB 或 AG-VPReID 9.6M frames, 团队有 OSS 传输能力); video person ReID 时间聚合很成熟(VID-Trans/PSTA/CTL/BiCnet/GRL/STMN/SINet); AG-VPReID 已 CVPR2025 占(三流网络+大数据集)。

问(**决定性 go/no-go**): 投入下载视频数据 + 做 video ReID **值不值得**?
① 有没有一个**真正新的视频 re-frame**——不是又一个时间聚合 / 质量帧加权(都被占), 而是 video / AG-VPReID 里被忽略的隐藏变量(例: tracklet 内身份证据的间歇可见性结构 / 帧间遮挡因果 / 航拍-地面视频特有的时空不对称采样), 能撑 CCF-B 方法稿 + 有**廉价(冻结帧特征零训练)kill-switch**?
② 还是视频也是成熟红海, 该诚实把 Hubness 写 analysis short 收尾、别再撞?
联网核查 video person ReID 2024-26 + AG-VPReID / AG-VPReID.VIR 占用 + temporal aggregation 现状。诚实: 给 **go(带具体新 re-frame + 廉价 kill-switch + 撞车核查)** 或 **no-go(+ 为什么 + Hubness analysis 收尾建议)**。务实中文, verdict + 信心 1-10。
codex
我会先核查近两年 video person ReID、AG-VPReID/VIR 相关论文与代码占位，再把“是否还有可打的方法空间”压成一个明确 verdict。重点会放在是否存在能零训练验证的隐藏变量，而不是泛泛列论文。
web search: 
web search: video person re-identification 2024 temporal aggregation transformer tracklet occlusion quality frame weighting person ReID survey 2025
web search: 
web search: site:arxiv.org video-based person re-identification 2024 2025 transformer temporal aggregation ReID
web search: 
web search: 2024 "video-based person re-identification" "temporal" "ReID" "CVPR"
web search: 
web search: https://arxiv.org/abs/2503.08121
web search: 
web search: 'Table 5' in https://arxiv.org/abs/2503.08121
web search: 
web search: https://arxiv.org/abs/2503.08121
web search: 
web search: https://arxiv.org/abs/2503.08121
web search: 
web search: https://ar5iv.org/html/2503.08121v2
web search: 
web search: https://arxiv.org/abs/2507.17995
web search: 
web search: https://arxiv.org/html/2507.17995
web search: 
web search: https://arxiv.org/html/2507.17995
web search: 
web search: https://arxiv.org/html/2506.22843
web search: 
web search: TF-CLIP Learning Text-Free CLIP for video-based person re-identification 2024
web search: 
web search: "A video is worth three views" "Trigeminal Transformers" video-based person re-identification 2024
web search: 
web search: "video-based person re-identification" "2026"
web search: 
web search: https://arxiv.org/abs/2601.01312
web search: 
web search: https://arxiv.org/abs/2601.05535
web search: 
web search: https://arxiv.org/abs/2601.05535
web search: 
web search: https://github.com/agvpreid25/AG-VPReID-Net
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
codex
我已经核查到两个关键事实：AG-VPReID/AG-VPReID.VIR 已经把三流、视角记忆、时序建模、RGB-IR、海拔分层都占了一轮；2026 的 VReID-XFD 又把极远距、物理元数据、形状先验、多粒度时序继续往前推了一步。现在我再快速对齐你仓库里的决策文档，避免把外部判断和你们已有 kill-switch 结论脱节。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/results.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/decisions.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/innovation_brainstorm.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/paper_materials/story.md: No such file or directory

codex
这个子目录下没有 `experiments/results.md` 这组相对路径，我会先定位实际实验文档位置；如果只是进入了 `experiments/...` 的深层目录，我会按实际路径读取同名文件。
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2

exec
/bin/zsh -lc ls in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
B_CONTAINMENT_DESIGN.md
GOPL_KILLSWITCH_DESIGN.md
HUBNESS_KILLSWITCH_DESIGN.md
OSAC_KILLSWITCH_DESIGN.md
SYNTHESIS_METHODOLOGY.md
all_papers.txt
analyses
batches
cvpb_containment_full.log
cvpb_containment_killswitch_design.md
debate
hubness_debate.sh
launch_30deep.sh
novelty_hubness.md
novelty_osac.md
ondisk_pivot.md
pivot
reassess
reassess.sh
reassess2
reassess2.sh
reviews
strategic_pivot.sh
validate
validate_candidates.sh
video_feasibility.md

exec
/bin/zsh -lc "rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
codex
仓库这块是 `litreview2` 的独立研究包，不是主实验目录。我会优先读 `video_feasibility.md`、`ondisk_pivot.md`、hubness/OSAC 复盘和方法论综合，作为最终判断的本地证据。
exec
/bin/zsh -lc "sed -n '1,260p' video_feasibility.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
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
session id: 019ef973-0b8e-7df2-9ba2-940180e6b5d3
--------
user
一个 ReID 团队连续 5 个 frozen-image/on-disk 方向被零训练 kill-switch 干净否掉(B航拍包含=前提错 / GOPL SMPL=occlusion-count代理 / Gallery Hubness=被k-reciprocal占 / OSAC谱坍缩=前提倒置训练反而升rank / RMA-TIReID text-image视觉流形锚定=边缘token-prototype≈color-only)。**frozen-feature + 现有 image/text on-disk 数据的 B 类方法空间穷尽。** 唯一真发现=gallery hubness 诊断(M(q)解释AP误差rho+0.60, analysis级非方法)。所有前序 codex 收敛: 唯一未测的 method-space 是**视频**(残酷终判推 AG-VPReID tracklet 时间证据 routing)。

但: 团队**无视频数据**(要下载 MARS ~4GB 或 AG-VPReID 9.6M frames, 团队有 OSS 传输能力); video person ReID 时间聚合很成熟(VID-Trans/PSTA/CTL/BiCnet/GRL/STMN/SINet); AG-VPReID 已 CVPR2025 占(三流网络+大数据集)。

问(**决定性 go/no-go**): 投入下载视频数据 + 做 video ReID **值不值得**?
① 有没有一个**真正新的视频 re-frame**——不是又一个时间聚合 / 质量帧加权(都被占), 而是 video / AG-VPReID 里被忽略的隐藏变量(例: tracklet 内身份证据的间歇可见性结构 / 帧间遮挡因果 / 航拍-地面视频特有的时空不对称采样), 能撑 CCF-B 方法稿 + 有**廉价(冻结帧特征零训练)kill-switch**?
② 还是视频也是成熟红海, 该诚实把 Hubness 写 analysis short 收尾、别再撞?
联网核查 video person ReID 2024-26 + AG-VPReID / AG-VPReID.VIR 占用 + temporal aggregation 现状。诚实: 给 **go(带具体新 re-frame + 廉价 kill-switch + 撞车核查)** 或 **no-go(+ 为什么 + Hubness analysis 收尾建议)**。务实中文, verdict + 信心 1-10。
codex
我会先核查近两年 video person ReID、AG-VPReID/VIR 相关论文与代码占位，再把“是否还有可打的方法空间”压成一个明确 verdict。重点会放在是否存在能零训练验证的隐藏变量，而不是泛泛列论文。
web search: 
web search: video person re-identification 2024 temporal aggregation transformer tracklet occlusion quality frame weighting person ReID survey 2025
web search: 
web search: site:arxiv.org video-based person re-identification 2024 2025 transformer temporal aggregation ReID
web search: 
web search: 2024 "video-based person re-identification" "temporal" "ReID" "CVPR"
web search: 
web search: https://arxiv.org/abs/2503.08121
web search: 
web search: 'Table 5' in https://arxiv.org/abs/2503.08121
web search: 
web search: https://arxiv.org/abs/2503.08121
web search: 
web search: https://arxiv.org/abs/2503.08121
web search: 
web search: https://ar5iv.org/html/2503.08121v2
web search: 
web search: https://arxiv.org/abs/2507.17995
web search: 
web search: https://arxiv.org/html/2507.17995
web search: 
web search: https://arxiv.org/html/2507.17995
web search: 
web search: https://arxiv.org/html/2506.22843
web search: 
web search: TF-CLIP Learning Text-Free CLIP for video-based person re-identification 2024
web search: 
web search: "A video is worth three views" "Trigeminal Transformers" video-based person re-identification 2024
web search: 
web search: "video-based person re-identification" "2026"
web search: 
web search: https://arxiv.org/abs/2601.01312
web search: 
web search: https://arxiv.org/abs/2601.05535
web search: 
web search: https://arxiv.org/abs/2601.05535
web search: 
web search: https://github.com/agvpreid25/AG-VPReID-Net
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
codex
我已经核查到两个关键事实：AG-VPReID/AG-VPReID.VIR 已经把三流、视角记忆、时序建模、RGB-IR、海拔分层都占了一轮；2026 的 VReID-XFD 又把极远距、物理元数据、形状先验、多粒度时序继续往前推了一步。现在我再快速对齐你仓库里的决策文档，避免把外部判断和你们已有 kill-switch 结论脱节。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/results.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/decisions.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/innovation_brainstorm.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/paper_materials/story.md: No such file or directory

codex
这个子目录下没有 `experiments/results.md` 这组相对路径，我会先定位实际实验文档位置；如果只是进入了 `experiments/...` 的深层目录，我会按实际路径读取同名文件。
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2

exec
/bin/zsh -lc ls in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
B_CONTAINMENT_DESIGN.md
GOPL_KILLSWITCH_DESIGN.md
HUBNESS_KILLSWITCH_DESIGN.md
OSAC_KILLSWITCH_DESIGN.md
SYNTHESIS_METHODOLOGY.md
all_papers.txt
analyses
batches
cvpb_containment_full.log
cvpb_containment_killswitch_design.md
debate
hubness_debate.sh
launch_30deep.sh
novelty_hubness.md
novelty_osac.md
ondisk_pivot.md
pivot
reassess
reassess.sh
reassess2
reassess2.sh
reviews
strategic_pivot.sh
validate
validate_candidates.sh
video_feasibility.md

exec
/bin/zsh -lc "rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
codex
仓库这块是 `litreview2` 的独立研究包，不是主实验目录。我会优先读 `video_feasibility.md`、`ondisk_pivot.md`、hubness/OSAC 复盘和方法论综合，作为最终判断的本地证据。

exec
/bin/zsh -lc "sed -n '1,220p' ondisk_pivot.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
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

exec
/bin/zsh -lc "sed -n '1,240p' SYNTHESIS_METHODOLOGY.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
# B 类 ReID 方法稿"创新构造方法论"综合(28 deep-codex 完整读, ~167 篇方法, 2026-06-24)

> 用户铁律: 不抄模块, 学**怎么把一个观察构造成能发的创新**。目标 CCF-B 方法稿。
> 本文 = 28 份 deep-codex(每份完整 Read, 0 截断)的横向综合 + 应用到我们自己资产。

## 一、通用配方(28 份全收敛, 无一例外)

1. **绝不先抛模块**, 先抓一个**具体失败 / 反直觉观察**(baseline 在哪崩、什么数字反常)。
2. **重定义**: 把失败改名成更尖锐的"隐藏变量 / 旧假设不成立"——"大家以为 X, 其实是 Y"。**这步最值钱、抄不来、是 novelty 的真正来源。**
3. **机制从重定义自然长出来**, 每个模块 ↔ 一个失败源(绑定越紧越好; 松了就是堆模块)。
4. **关键证据证"重定义对"**(不是证"机制涨点"): 一个**如果失败就推翻叙事**的诊断指标/消融/可视化。
5. **reviewer 买视角不买模块**。模块可以全是旧零件(GNN/attention/OT/CLIP), 只要被一个新问题串住。

## 二、20+ 个"重定义动作"catalog(我们的招式库)

| # | 动作 | 样本 |
|---|------|------|
| 1 | 隐藏变量 | DMDL 模态偏置"从数据传到标签传到特征"(因果图); DMPF 模态+姿态双因子 |
| 2 | **把问题数学化** | GAReID: 相似度=所有 part-pair 平均, 错配对 >> 对齐对, 一个公式解释 misalignment |
| 3 | 可测中间变量 | 梯度一致性; MDRR 模态分歧降低率; NCC(C→X) 因果强度 |
| 4 | 物理/常识约束 | GSTNET 地理可达性(5秒跨20km不可能=图边) |
| 5 | 偏差诊断 | EAIBC 颜色过度依赖; MSP 发型捷径 |
| 6 | **"太晚了"/用错位置** | HCCL 噪声在前向传播就污染了; Pose-Skeleton 遮挡信息扩散→中间层拦; training-free TI-ReID 图库结构当**测试期先验**("Prototype-in-Training 反而掉点") |
| 7 | **改信号的角色** | 衣服文本不识别人而是"告诉模型压制什么"; LVLM→身份语义token; prompt→可学习中间模态 |
| 8 | 数据中心反直觉 | 车辆从"噪声该丢"→"positive-incentive noise" |
| 9 | per-pair 最优条件 | 每对图像有自己的最优光照距离, 最优不一定是正常光照 |
| 10 | **非对称包含** | PDA: 文本分布⊇图像分布; 方差=语义范围(非噪声) |
| 11 | 回收"该丢的" | 形状不是噪声(红外形状估计错误才是); 噪声样本历史自校正 |
| 12 | **表示形态错了** | 3D Gait: 稀疏SMPL参数没法和稠密appearance融→蒸成稠密时空场; "换辅助模态"消融证 dense>skeleton>SMPL参 |
| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
| 14 | 隔离混杂变量(负结论也发) | Rethinking Joint Opt: 联合优化的收益其实来自尾部解析 |
| 15 | **修训练组织非模型** | curriculum CC-ReID: 先单衣后逐步加最难衣, 不改模型不加输入 |
| 16 | "顺序错了" | Two-stage KD: 先同模态收紧再跨模态 |
| 17 | **因果(最强)** | P(Y\|X)→P(Y\|do(X)) backdoor 切断衣服捷径 |
| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
| 19 | 旧法在新基座失效 | CLIP 细粒度: 朴素局部切分伤预训练空间("朴素 baseline 比 global 还差") |
| 20 | **对齐伤判别性** | CycleTrans neutral-yet-discriminative; BDLF base/detail; 多频身份线索 |
| 21 | 目标函数错配 | CAP: 直接优化跨模态 AP |
| 22 | 数字驱动 | CLNS: 跨相机正样本距 1.06 > 同相机 0.78 → 相机结构噪声 |

**证"重定义对"的硬手段**: 数学分解 / 失败检索样例 / 可测诊断指标 vs per-query AP 相关 / 互补性 Venn / 伪标签质量曲线 / 前后 attention map / "替换机制"破坏性对照 / 参数曲线(证不是越多越好)。

## 三、应用到我们自己(候选 re-framing, 每个带廉价 kill-switch)

我们独有资产: CARGO/AG-ReID.v2 极端跨视角(航拍↔地面 90°, 航拍低清俯视); **观察: avg-pool 52.37 > token-MaxSim 45.19(差7分)**; SMPL 几何基建; Swin 67.33(backbone)。

> ⚠️ 用户已打掉"MaxSim<avg→局部不可靠→用avg"(推到底=零贡献)。**新候选的硬门槛: 机制必须 beat avg, 不能退化成 avg。** aerial-ground 几何/可见性是红海(避开)。

### 候选 B(主推): 非对称包含 — 把对称匹配换成"航拍证据⊆地面证据"
- **观察**: 航拍低清俯视 = 信息**欠定**(看不清脸/纹理); 地面高清正面 = 信息**确定**。但所有 cross-view ReID 用**对称** cosine/MaxSim 匹配, 默认两视角信息对等。
- **重定义(move 10+22)**: "大家以为跨视角是对称对齐问题, 其实航拍↔地面是**非对称信息包含**: 航拍身份证据是一个**范围(宽分布)**, 应被地面的**窄分布包含**。" 方差=该视角的信息欠定度(非噪声)。
- **机制(自然长出)**: 每张图建成分布(均值+方差), 航拍方差大/地面方差小; 包含损失让航拍分布⊆地面分布(非对称, 不是拉近两个点)。
- **证重定义**: ① 航拍图特征方差是否系统性 > 地面(分布可视化); ② 非对称包含距离是否同时 beat 对称cosine 和 MaxSim; ③ 只在高视角差样本收益更大。
- **切开(避红海)**: vs PDA(文本-图像) = 我们是 cross-VIEW; vs OT-ReID(CM-EMD/G2DA) = 我们的非对称方向(航拍⊆地面)由**成像物理**(俯视低清欠定)定, 不是纯视觉 cost。
- **kill-switch(零训练)**: frozen Swin(swin_fix256, 67.33)提特征, CARGO A↔G: 对称cosine vs 非对称包含距离(航拍当宽高斯/地面窄高斯, 用马氏或KL包含)。包含明显赢 → re-framing 有腿; 打平 → 死, 回头。

### 候选 C: 对齐伤判别性(move 20, CycleTrans 思路迁到 cross-view)
- **观察**: avg>MaxSim 说明强行局部对齐(MaxSim)在极端跨视角**有害**。
- **重定义**: "航拍和地面不共享可对齐的**局部**, 只共享**全局身份**+**视角特有**判别线索; 强行对齐(MaxSim)塌掉视角特有线索, 均匀平均(avg)稀释它。" → 学"视角中性但保判别"特征。
- **风险**: CycleTrans/BDLF 已做 neutral-yet-discriminative(cross-modal), 切开点是 cross-view; 较弱, 当候选 B 的备胎。
- **kill-switch**: frozen Swin, 中性特征(去视角分量)+视角特有残差 分开重组 vs avg。

### 候选 D: 因果(move 17, 最强范式但需想清 confounder)
- **重定义**: "cross-view ReID 普通训练学 P(Y|view-entangled X), 视角是 confounder; 应学 P(Y|do(view)) 用 backdoor 对所有视角求和切断'视角-身份'伪相关。"
- **风险**: Causal CC-ReID 已用 backdoor(confounder=衣服); 我们 confounder=视角, 切开点要硬。需想清 confounder dictionary 怎么建(视角桶?)。
- **kill-switch**: NCC(view→X) 因果强度, do-intervention 前后 per-view ID 可分性。

### 候选 E(弱, 记录): 表示形态 + SMPL view-canonical
- "Beyond geometry"(deep_3#6)已做 UVTexture canonical-viewpoint for aerial-ground(AG-ReID.v2)→ **直接撞车**, 降优先级。

## 四、下一步(用户授权: 拿不定和 codex 讨论)
启 4-5 个 codex(--search)对候选 B/C/D 做: ① novelty/撞车检索(PDA/OT-ReID/CycleTrans/causal-ReID 边界); ② 哪个最强 + 为什么; ③ kill-switch 设计是否真能证伪。**绝不动手前先查 novelty + 必须有廉价 kill-switch(铁律)。**

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

web search: 
web search: 'three-view network' in https://arxiv.org/html/2503.08121
web search: 
web search: 'AG-VPReID.VIR' in https://arxiv.org/html/2507.17995
web search: 
web search: "Video-based Person Re-identification" "survey" "temporal" "2024"
web search: 
web search: "A video is worth three views" Trigeminal Transformers ReID
web search: 
web search: "PSTA" "Video-based Person Re-identification" pose-guided spatial temporal attention
web search: 
web search: "Pose-guided Spatial-Temporal Alignment" video-based person re-identification PSTA
web search: 
web search: PSTA ReID pose-guided person re-identification video
web search: 
web search: "VID-Trans" "video-based person re-identification"
web search: 
web search: "video person re-identification" "2024" "MARS" "rank-1" "mAP"
web search: 
web search: G2A-VReID 2024 aerial ground video person re-identification dataset method temporal
web search: 
web search: 'benchmark' in https://arxiv.org/abs/2601.01312
web search: 
web search: 'multi-granularity' in https://arxiv.org/abs/2601.05535
web search: 
web search: 'Dataset' in https://github.com/agvpreid25/AG-VPReID-Net
web search: 
web search: "intermittent visibility" "person re-identification" video tracklet
web search: 
web search: "temporal occlusion" "video-based person re-identification"
web search: 
web search: "quality" "frame weighting" "video-based person re-identification"
web search: 
web search: "Quality Aware Network" "video-based person re-identification"
web search: 
web search: https://arxiv.org/abs/2303.11332
web search: 
web search: https://arxiv.org/abs/2312.09627
web search: 
web search: https://arxiv.org/abs/2408.07500
codex
**Verdict: no-go。信心 8/10。**

不建议投入 AG-VPReID 9.6M frames 或完整 video ReID 方法线。最多允许一个 1-2 天、严格封顶的 MARS 小下载 sanity check；但它不应被视为新主线。

理由很硬：

1. **video temporal aggregation 已经不是空地**  
   近年线索已经覆盖 temporal attention、spatial-temporal graph、distractor/occlusion memory、CLIP memory、multi-granularity temporal modeling。比如 STMN 已明确处理视频中的 partial occlusion / temporal distractors，TF-CLIP 已用 sequence memory + temporal memory diffusion 做视频表征，CTL/GRL/TMT 也都在时空关系、局部线索、跨视图聚合上占过。

2. **AG-VPReID 空间被 dataset paper 自己占得很满**  
   AG-VPReID 是 CVPR 2025，6,632 IDs / 32,321 tracklets / 9.6M frames，方法已经是三流：Temporal-Spatial、Normalized Appearance、Multi-Scale Attention，显式处理 temporal discontinuity、resolution/viewpoint、altitude scale。官方 repo 也把 altitude 分层性能、A2G/G2A 协议都列出来了。  
   来源：AG-VPReID paper / repo：6,632 IDs、32,321 tracklets、9.6M frames、三流 AG-VPReID-Net。  
   https://arxiv.org/html/2503.08121  
   https://github.com/agvpreid25/AG-VPReID-Net

3. **AG-VPReID.VIR 又把 cross-platform + RGB-IR + memory + temporal 占了一轮**  
   AG-VPReID.VIR 是 IJCB 2025，1,837 IDs / 4,861 tracklets / 124,855 frames，TCC-VPReID 也是三流：style-robust、memory-based cross-view、intermediary-guided temporal。也就是说“跨平台隐藏变量 + temporal + memory routing”已经非常接近被占。  
   https://arxiv.org/html/2507.17995  
   https://github.com/agvpreid25/AG-VPReID.VIR

4. **2026 VReID-XFD 继续压缩“航拍-地面视频新问题”的空间**  
   VReID-XFD 已经把 extreme far distance、altitude/distance/nadir view、physical metadata 做成 challenge；第一名 SAS-VPReID 用 CLIP memory、多粒度 temporal modeling、shape dynamics。这个方向已经进入 challenge-工程竞争。  
   https://arxiv.org/abs/2601.01312  
   https://arxiv.org/abs/2601.05535

5. **你提出的唯一像样 re-frame 会被 reviewer 归类成旧问题**  
   “tracklet 内身份证据间歇可见性 / temporal evidence routing”听起来比 frame weighting 高一级，但没有额外标注或新协议时，落地会被看成：
   - temporal attention
   - quality frame selection
   - occlusion/distractor-aware memory
   - multi-granularity temporal modeling  
   这些都已有强先例。零训练 kill-switch 即使成功，也大概率只能证明“best segment / sparse frame > average pooling”，这不足以撑 B 类方法稿。

**唯一可做的廉价 kill-switch，不建议转主线**

若团队心理上必须最后验一次，只下载 MARS，做 frozen frame feature：

- tracklet mean pooling vs top-k frame pooling vs segment max pooling
- “oracle best-frame gap”是否显著解释失败 query
- intermittent evidence score 是否比简单 frame quality / blur / occlusion count / similarity variance 更能解释 AP

kill 条件：如果 improvement 不能明显超过简单 top-k / quality weighting，或相关性不能超过现有 hubness M(q)，立刻终止。不要下载 AG-VPReID 全量。

**建议收尾方向**

把 Hubness 写成 analysis short：  
核心叙事是“强 ReID 的残差失败不是 pairwise similarity，而是 gallery topology 的 many-to-one 吸附”。主结果放 `M(q)` 解释 AP error，rho +0.60；然后诚实写 remedy 被 k-reciprocal / re-ranking 吞掉，因此这是诊断贡献，不是训练方法贡献。

一句话：**video 不是完全没现象，而是你们能廉价拿到的现象不足以避开成熟 temporal aggregation + AG-VPReID 系列占位。别再撞，收 Hubness analysis 更理性。**
tokens used
189,763
**Verdict: no-go。信心 8/10。**

不建议投入 AG-VPReID 9.6M frames 或完整 video ReID 方法线。最多允许一个 1-2 天、严格封顶的 MARS 小下载 sanity check；但它不应被视为新主线。

理由很硬：

1. **video temporal aggregation 已经不是空地**  
   近年线索已经覆盖 temporal attention、spatial-temporal graph、distractor/occlusion memory、CLIP memory、multi-granularity temporal modeling。比如 STMN 已明确处理视频中的 partial occlusion / temporal distractors，TF-CLIP 已用 sequence memory + temporal memory diffusion 做视频表征，CTL/GRL/TMT 也都在时空关系、局部线索、跨视图聚合上占过。

2. **AG-VPReID 空间被 dataset paper 自己占得很满**  
   AG-VPReID 是 CVPR 2025，6,632 IDs / 32,321 tracklets / 9.6M frames，方法已经是三流：Temporal-Spatial、Normalized Appearance、Multi-Scale Attention，显式处理 temporal discontinuity、resolution/viewpoint、altitude scale。官方 repo 也把 altitude 分层性能、A2G/G2A 协议都列出来了。  
   来源：AG-VPReID paper / repo：6,632 IDs、32,321 tracklets、9.6M frames、三流 AG-VPReID-Net。  
   https://arxiv.org/html/2503.08121  
   https://github.com/agvpreid25/AG-VPReID-Net

3. **AG-VPReID.VIR 又把 cross-platform + RGB-IR + memory + temporal 占了一轮**  
   AG-VPReID.VIR 是 IJCB 2025，1,837 IDs / 4,861 tracklets / 124,855 frames，TCC-VPReID 也是三流：style-robust、memory-based cross-view、intermediary-guided temporal。也就是说“跨平台隐藏变量 + temporal + memory routing”已经非常接近被占。  
   https://arxiv.org/html/2507.17995  
   https://github.com/agvpreid25/AG-VPReID.VIR

4. **2026 VReID-XFD 继续压缩“航拍-地面视频新问题”的空间**  
   VReID-XFD 已经把 extreme far distance、altitude/distance/nadir view、physical metadata 做成 challenge；第一名 SAS-VPReID 用 CLIP memory、多粒度 temporal modeling、shape dynamics。这个方向已经进入 challenge-工程竞争。  
   https://arxiv.org/abs/2601.01312  
   https://arxiv.org/abs/2601.05535

5. **你提出的唯一像样 re-frame 会被 reviewer 归类成旧问题**  
   “tracklet 内身份证据间歇可见性 / temporal evidence routing”听起来比 frame weighting 高一级，但没有额外标注或新协议时，落地会被看成：
   - temporal attention
   - quality frame selection
   - occlusion/distractor-aware memory
   - multi-granularity temporal modeling  
   这些都已有强先例。零训练 kill-switch 即使成功，也大概率只能证明“best segment / sparse frame > average pooling”，这不足以撑 B 类方法稿。

**唯一可做的廉价 kill-switch，不建议转主线**

若团队心理上必须最后验一次，只下载 MARS，做 frozen frame feature：

- tracklet mean pooling vs top-k frame pooling vs segment max pooling
- “oracle best-frame gap”是否显著解释失败 query
- intermittent evidence score 是否比简单 frame quality / blur / occlusion count / similarity variance 更能解释 AP

kill 条件：如果 improvement 不能明显超过简单 top-k / quality weighting，或相关性不能超过现有 hubness M(q)，立刻终止。不要下载 AG-VPReID 全量。

**建议收尾方向**

把 Hubness 写成 analysis short：  
核心叙事是“强 ReID 的残差失败不是 pairwise similarity，而是 gallery topology 的 many-to-one 吸附”。主结果放 `M(q)` 解释 AP error，rho +0.60；然后诚实写 remedy 被 k-reciprocal / re-ranking 吞掉，因此这是诊断贡献，不是训练方法贡献。

一句话：**video 不是完全没现象，而是你们能廉价拿到的现象不足以避开成熟 temporal aggregation + AG-VPReID 系列占位。别再撞，收 Hubness analysis 更理性。**
