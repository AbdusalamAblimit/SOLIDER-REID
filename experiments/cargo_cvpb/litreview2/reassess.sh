#!/bin/bash
# 3-codex 战略重评: B+GOPL 连续死后, 残酷否决/找全新方向/最后救遮挡, --search
OUT=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess
mkdir -p "$OUT"
PX="HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1"
CODEX=/opt/homebrew/bin/codex

CTX="一个做行人 ReID 的团队要发 CCF-B **方法稿**, 刚连续用零训练 kill-switch 干净杀掉两个方向:
- **死1**: 航拍-地面'物理定向不确定性包含'——前提 σ_aerial<σ_ground(航拍其实更平滑)就错, 包含'收益'是检索 artifact。
- **死2(刚)**: GOPL'正样本关系粒度错了'(用 SMPL 共同可见人体表面 overlap 当 same-ID 正边可靠性)。冻结最强 occluded_duke ckpt(exp255, sanity mAP73.05≈训练73.2)测: cosine 距离确实弱负相关 overlap(rho≈-0.13)但: D5 随机关节子集打平(-0.1294≈真-0.1273)、cov2d 控住可见关节数后归零(+0.0145)、#vis 关节数 baseline 反而更强(-0.1321)、相机对(视角差)解释方差 eta²0.062 比 SMPL cov 0.018 还大、cov3d(3D自遮挡)没明显强于 cov2d → 隐藏变量是'遮挡越重越难'(occlusion-count)不是'共同可见表面', SMPL 几何=VPM/QPM/occlusion-count 换名无独特信号。

**残酷 meta-pattern**: 团队'独有资产'SMPL/人体几何**反复 reduce to 廉价代理**——exp333 SMPL-β≈random、pose 融 CLIP 五角度全死、SMPL-anchor VI-ReID 死、GOPL cov≈occlusion-count+view-diff。**SMPL 对 ReID 没有超出 occlusion-count/视角的独特判别信号。** 团队三大资产全弱: ①SMPL/几何(反复证伪) ②遮挡 ReID(历史5死角+三堵墙 completion/occluder-gate/visibility, exp109 oracle headroom 是 identity-conditioned 不可实现的墙) ③航拍-地面(ViSA/GSAlign/SeCap/DTST 占满+转 video/challenge)。**真的硬资产只有强 Swin/SOLIDER backbone**(occluded_duke 73.2 / CARGO 67.3)但那是 backbone 非方法。可用数据: occluded_duke / market / MSMT17 / CARGO / AG-ReID.v2 / occluded_reid。

方法论(团队读167篇方法稿总结): B类创新=失败观察→重定义隐藏变量→机制自然长出→证重定义对。22招式(数学化/可测中间变量/对齐伤判别/因果/改信号角色/顺序错了/新协议/非对称包含/数据中心反直觉/旧法在新基座失效等)。"

declare -a ROLES
ROLES[1]="角色=**残酷判官**。不留情面: 连续2个cheap-kill负 + SMPL反复证伪 + 三大资产全弱, 这个团队到底**还有没有**能发B类方法稿的真路? 还是诚实答案='手里的empirical资产出不了干净B类method, 该换问题域/换数据/换打法'? 联网核查ReID 2024-2026 method空间。**别和稀泥**: 要么明确指1条最可能成的(带理由+廉价kill-switch), 要么明确说'没有, 该转去X'(X具体)。"
ROLES[2]="角色=**全新方向探子**。彻底放下SMPL/遮挡/航拍-地面三个失败区(不准碰)。团队只剩: 强Swin/SOLIDER backbone(可冻结当强特征源) + occluded_duke/market/MSMT/CARGO数据 + 22招式 + 读过167篇。用联网+招式提**1-2个真正新的B类方向**: 新问题定义或新观察(不是新模块), 每个带触发观察/重定义/机制怎么长/**廉价(最好零训练)kill-switch**/撞车核查。优先'用强backbone冻结特征就能验隐藏变量'的方向(像我们之前那种零训练kill-switch)。务实中文。"
ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"

for i in 1 2 3; do
  nohup env ${PX} ${CODEX} --search exec -s read-only --color never "${CTX}

== 你的任务 ==
${ROLES[$i]}" > "$OUT/r_${i}.md" 2>&1 &
  echo "launched reassess-codex ${i} (PID $!)"
  sleep 2
done
echo "=== 3 个重评 codex 启动 ==="
