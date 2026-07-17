#!/bin/bash
# 3-codex 战略 panel: B containment FAIL 后, 救援/转向/残酷否决, --search
OUT=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
mkdir -p "$OUT/pivot"
PX="HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1"
CODEX=/opt/homebrew/bin/codex

CTX="背景: 一个做航拍-地面行人ReID(CARGO/AG-ReID.v2)的团队, 要发CCF-B**方法稿**。刚用零训练kill-switch干净杀掉了最被看好的方向。

**死掉的方向B(航拍-地面物理定向不确定性包含)**: 重定义='跨视角不是对称匹配, 而是航拍低清=信息欠定(宽分布)应被地面窄分布非对称包含'。零训练kill-switch(冻结swin_fix256, cosine A→G 67.41≈训练67.33 sanity过, codex审脚本approve)三假设全证伪:
- ①σ_aerial < σ_ground(航拍q156.96/g167.47 < 地面q171.64/g172.81双侧; 合成退化σ反而降115.41<118.93)。'航拍更欠定=宽分布'前提**错的**——航拍低清=少高频细节=更平滑=低TTA方差。
- ②包含'收益'+1.2(KL68.62 vs cosine67.41)是假象: equal-var Maha(σ-free)也67.94>cosine, 所有对称分布距离(sym-KL56/JS55/Bhatt44)远低于cosine。非对称方向是检索artifact(赢的KL只是均值项除以query方差; G→A CORRECT崩到17.37因除gallery方差)。
- ③image-level σ无用(C3 view-mean 69.07不降反升/C4同视角置换67.47/C5 hardness 66.63 都≈correct不掉)。

**团队CARGO empirical资产盘点(残酷)**: avg-pool 52.37(唯一扎实观察, 但OVLI/OVP/MaxSim/containment各种re-frame反复reduce to avg或死)/ OVP 50.11(撞CMPC)/ token-MaxSim 45.19(死, 不如avg)/ Swin port 67.33(backbone非方法)/ SMPL人体几何基建(mesh/joints/2D投影, 但exp333证SMPL-β对ReID≈random)。历史: 遮挡ReID 5个角度全死后才转的这条empirical线。

**方法论(团队刚读167篇方法稿总结)**: B类创新=先抓失败观察→重定义'大家以为X其实Y'隐藏变量→机制自然长出→证重定义对。22招式: 数学化/可测中间变量/对齐伤判别/因果/表示形态错/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉 等。红海(避开): 航拍-地面几何对齐/可见性/纯benchmark(GSAlign/VDT/SeCap/AG-VPReID/ViSA已占)。"

declare -a ROLES
ROLES[1]="角色=**救援者**。给定新发现 σ_aerial<σ_ground(航拍更平滑/少细节, 地面细节丰富)+ avg>MaxSim 这两个硬事实, 用联网搜索找一个**还活着的 B 类 re-frame**, 必须同时满足: (a)符合'航拍平滑低细节'而非'航拍噪声/欠定'; (b)不 reduce to avg(机制不能退化成平均池化); (c)有廉价零训练 kill-switch; (d)不撞 GSAlign/VDT/SeCap/ViSA/cross-resolution 红海。逐个候选查 novelty。如果找不到符合的, 诚实说'救不动'。务实中文, 给候选+kill-switch+撞车核查。"
ROLES[2]="角色=**转向者**。假设 avg>MaxSim 这个 hook 是死胡同(反复 reduce to avg)。团队资产=SMPL人体几何基建 + Swin/SOLIDER backbone + CARGO/AG-ReID.v2 数据 + 遮挡/VI-ReID 历史。用 22 招式 + 联网, 提 1-2 个**换问题**的 B 类方向(可以离开 avg>MaxSim 甚至离开纯 aerial-ground), 每个带: 触发观察/重定义/机制怎么长/廉价 kill-switch/撞车核查。重点用团队独有的 SMPL 几何当差异化(但记住 exp333 证 SMPL-β≈random, 别再走那条)。务实中文。"
ROLES[3]="角色=**残酷否决者**。不留情面判断: 整个 CARGO/aerial-ground empirical 方向, 对一篇 B 类**方法稿**(不是 benchmark/不是 backbone), 是不是已经是死胡同?证据: 遮挡5死角→转empirical→OVLI/OVP/MaxSim/containment 又全倒, method 始终不成形, 只有 avg 这个平凡 baseline 撑着。联网核查 aerial-ground ReID 2024-2026 现状(还有没有 method 空间, 还是已被 GSAlign/VDT/AG-VPReID 这代占满)。如果是死胡同, 明确说'该放弃换战场', 并指出团队最该回到哪类问题(基于它的真实资产: SMPL几何/Swin/遮挡历史)。如果不是死胡同, 指出唯一还值得赌的点。务实中文, 别和稀泥。"

for i in 1 2 3; do
  nohup env ${PX} ${CODEX} --search exec -s read-only --color never "${CTX}

== 你的任务 ==
${ROLES[$i]}" > "$OUT/pivot/p_${i}.md" 2>&1 &
  echo "launched pivot-codex ${i} (PID $!)"
  sleep 2
done
echo "=== 3 个战略 codex 启动 ==="
