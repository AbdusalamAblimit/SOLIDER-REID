#!/bin/bash
# 20-codex 论文库创新挖掘: 读 167 篇 ReID B类论文摘要+intro, 提创新套路 + 生成团队资产强创新候选
LIB=/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
OUT=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview
mkdir -p "$OUT/batches" "$OUT/reviews"
cd "$LIB" || exit 1
ls *.pdf > "$OUT/all_papers.txt"
rm -f "$OUT/batches/"b* 2>/dev/null
split -l 9 "$OUT/all_papers.txt" "$OUT/batches/b"

PX="HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1"
CODEX=/opt/homebrew/bin/codex

i=0
for B in "$OUT/batches/b"*; do
  i=$((i+1))
  PAPERS=$(cat "$B")
  PROMPT="你是 ReID 论文创新挖掘员 ${i}/20。当前目录就是论文库,下面这批 PDF 在本目录。用 \`pdftotext -l 3 '文件名' -\` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(${i}/20):
${PAPERS}

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。"
  nohup env ${PX} ${CODEX} --search exec -s read-only --color never "${PROMPT}" > "$OUT/reviews/lit_${i}.md" 2>&1 &
  echo "launched lit-codex ${i} (PID $!)"
  sleep 2
done
echo "=== 20 lit-codex 全部启动 ==="
