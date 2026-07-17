#!/bin/bash
OUT=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/explore20
mkdir -p "$OUT"
PX="HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1"
CODEX=/opt/homebrew/bin/codex
CTX="一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。"
declare -a D
D[1]="车辆ReID(Vehicle ReID, VeRi-776/VehicleID/VERI-Wild)"
D[2]="换衣ReID(Cloth-Changing, PRCC/LTCC/DeepChange): 身份线索从外观转体型/步态/脸"
D[3]="终身/持续ReID(Lifelong/Continual): 增量学新域不忘旧"
D[4]="可见光-红外跨模态ReID(VI-ReID, SYSU-MM01/RegDB): **避开团队已死的SMPL-anchor**, 找别的模态gap切口"
D[5]="文本-图像ReID(Text-to-image, CUHK-PEDES/RSTPReid): **超出团队已死的token-prototype锚定(RMA)**, 别的切口"
D[6]="群组ReID(Group ReID): 多人整体匹配/成员置换/布局"
D[7]="无监督/域自适应ReID(USL/UDA, 伪标签/聚类)"
D[8]="低/跨分辨率ReID(Low/Cross-Resolution)"
D[9]="开集/开放世界ReID(Open-set/Open-world): query可能不在gallery/拒识"
D[10]="长尾/不均衡ReID(Long-tail)"
D[11]="噪声标签ReID(Noisy-label/Label-noise)"
D[12]="域泛化ReID(DG-ReID, 多源训练→未见域直测)"
D[13]="测试时自适应/在线ReID(Test-time adaptation/Online)"
D[14]="基础模型时代ReID(CLIP/DINO/SAM-based): **避开团队已死的通用FM-import**, 找FM别的用法"
D[15]="多模态ReID(RGB-D/sketch草图查询/3D点云)"
D[16]="隐私保护/联邦ReID(Privacy/Federated)"
D[17]="对抗鲁棒ReID(Adversarial-robust/物理攻击防御)"
D[18]="步态/骨架ReID(Gait/Skeleton-based): **避开团队已死的SMPL几何**, 用2D骨架序列/步态别切口"
D[19]="动物ReID/细粒度检索迁移(Animal-ReID/fine-grained, 跨物种或行人迁动物商品)"
D[20]="★主动构造新问题/新协议(construct novel setting): 最自由——定义文献还没有的ReID部署场景/评测协议/新约束(贴真实部署痛点), 让方法自然长出"
for i in $(seq 1 20); do
  nohup env ${PX} ${CODEX} --search exec -s read-only --color never "${CTX}

== 你负责的方向 ==
${D[$i]}

== 输出 ==
找 1-2 个该领域/情况下能撑 CCF-B 方法稿的**具体机会**: 触发观察或失败 → 重定义(隐藏变量/新问题) → 机制草案 → **廉价kill-switch(最好零训练)** → 撞车核查(联网 2024-26, 不能红海)。诚实: 若该领域也红海/无廉价入口, 直说。每个机会给 verdict + 信心1-10 + 团队现有数据(market/MSMT/occluded_duke/CARGO/AG-ReID.v2/RSTPReid)能否首验还是需新数据。务实中文。" > "$OUT/d_${i}.md" 2>&1 &
  echo "launched explore-codex ${i} (PID $!): ${D[$i]:0:36}"
  sleep 1
done
echo "=== 20 explore codex 启动 ==="
