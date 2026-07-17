#!/bin/bash
# 4-codex 红蓝裁判 panel: 验证我们的 B 类 re-framing 候选, --search 联网查 novelty/撞车
OUT=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
mkdir -p "$OUT/validate"
PX="HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1"
CODEX=/opt/homebrew/bin/codex

CTX="背景: 我们做**航拍-地面跨视角行人 ReID**(数据集 CARGO / AG-ReID.v2, 视角差约 90 度, 航拍图低分辨率俯视、地面图高清正面)。目标发 **CCF-B 方法稿**(不是 backbone 结果, 不是堆模块)。
我们手里唯一扎实的观察: 在 CARGO 航拍↔地面上, **普通 avg-pool 全局特征(mAP 52.37) 明显打败 token-set MaxSim late-interaction(ColBERT/AlignedReID 式局部匹配, 45.19), 差 7 分**。我们还有 SMPL 人体几何基建(mesh/joints/2D投影)、一个 Swin backbone(冻结提特征 mAP 67.33, 但这是 backbone 不是方法)。
我们刚读完 ~167 篇 ReID 方法稿, 总结出 B 类创新配方: **先抓失败观察 → 重定义成'大家以为X其实是Y'的隐藏变量 → 机制从重定义自然长出 → 关键证据证'重定义对'而非'机制涨点'。** 红海(避开): 航拍-地面的几何对齐/可见性/纯 benchmark(GSAlign/AG-VPReID/multi-platform/VDT 已占)。

我们的候选 re-framing:
[B 主推] **非对称信息包含**: 航拍低清俯视=信息欠定(看不清脸/纹理), 地面高清正面=信息确定; 但所有 cross-view ReID 用**对称** cosine/MaxSim 匹配。重定义='跨视角不是对称对齐, 而是航拍身份证据=一个范围(宽分布)应被地面的窄分布**非对称包含**', 方差=该视角信息欠定度。机制: 每图建分布(均值+方差), 包含损失让航拍分布⊆地面分布。kill-switch(零训练): 冻结 Swin 提特征, 对称cosine vs 非对称包含距离(航拍宽高斯/地面窄高斯, 马氏或KL包含) on CARGO A↔G。
[C 备胎] **对齐伤判别性**: avg>MaxSim 说明强行局部对齐在极端跨视角有害; 重定义='航拍地面不共享可对齐局部, 只共享全局身份+视角特有判别线索, MaxSim塌掉它/avg稀释它'→学视角中性但保判别特征。
[D 范式] **因果**: 普通训练学 P(Y|视角纠缠的X), 视角是 confounder; 应学 P(Y|do(视角)) 用 backdoor 切断'视角-身份'伪相关。"

declare -a ROLES
ROLES[1]="你的角色=**蓝队杀手**。用联网搜索**尽全力杀死候选 B(非对称包含 for 跨视角)**: 找出让它不新颖的确切先例——重点查 PDA/Probabilistic Distribution Alignment(文本-图像包含)、probabilistic/Gaussian embedding ReID、OT-based ReID(CM-EMD/G2DA/CVFT)、set-based/distribution matching ReID、aerial-ground(VDT/GSAlign/AG-VPReID/SDPL)、cross-resolution ReID。逐个判断它们是否已经做了'非对称分布包含'或'低清query信息欠定建模'。如果杀不死, 明确说它靠什么活下来(切开点)。务实中文, 给 verdict(撞车/存活)+ 证据链接。"
ROLES[2]="你的角色=**红队辩护**。为候选 B 辩护它是最强 B 类赌注: 用联网把它和 PDA/OT-ReID/probabilistic-embedding 的**切开点**打磨锋利(我们是 cross-VIEW 不是 text-image; 非对称方向航拍⊆地面由**成像物理**定不是纯视觉cost)。设计**最能打动 reviewer 的杀手级证据**(哪个消融/可视化/诊断指标一旦失败就推翻叙事)。再判断这个 idea 够不够一篇 B 类的体量。务实中文, 给信心分 1-10 + 证据设计。"
ROLES[3]="你的角色=**独立裁判**。不预设立场, 用联网核查后**独立给 B/C/D 排序**, 并且——关键——基于我们那个 avg>MaxSim 观察, 看看有没有**比 B/C/D 更强的全新 re-frame**(用我们总结的 22 个重定义动作: 数学化/可测中间变量/对齐伤判别/因果/表示形态/改信号角色/顺序错了 等)。给出你认为最该做的 1 个, 和它的廉价零训练 kill-switch。务实中文。"
ROLES[4]="你的角色=**kill-switch 批判员**。只盯一件事: 候选 B/C/D 的'零训练冻结Swin' kill-switch **是否真能证伪 re-framing**? 比如候选 B 的'非对称包含距离 beat 对称cosine'——会不会赢是因为别的混杂原因(比如只是 query/gallery 归一化差异、或方差只是反映难度而非信息欠定)? 怎么设计**对照**才能干净隔离'非对称包含'这个机制本身的功劳(参考方法论里的'替换机制破坏性对照')? 给出加固后的 kill-switch 协议。务实中文。"

for i in 1 2 3 4; do
  nohup env ${PX} ${CODEX} --search exec -s read-only --color never "${CTX}

== 你的任务 ==
${ROLES[$i]}" > "$OUT/validate/v_${i}.md" 2>&1 &
  echo "launched validate-codex ${i} (PID $!)"
  sleep 2
done
echo "=== 4 个验证 codex 全部启动 ==="
