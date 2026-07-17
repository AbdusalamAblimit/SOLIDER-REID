#!/bin/bash
# 30-codex 深读: 读 167 篇完整方法部分, 学"怎么创新"的方法论(非抄模块), --search 全开
LIB=/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
OUT=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
mkdir -p "$OUT/batches" "$OUT/reviews"
cd "$LIB" || exit 1
ls *.pdf > "$OUT/all_papers.txt"
rm -f "$OUT/batches/"b* 2>/dev/null
split -l 6 "$OUT/all_papers.txt" "$OUT/batches/b"   # 167/6 ~= 28 批, 每批 ~6 篇

PX="HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1"
CODEX=/opt/homebrew/bin/codex

i=0
for B in "$OUT/batches/b"*; do
  i=$((i+1))
  PAPERS=$(cat "$B")
  PROMPT="你是 ReID 创新方法论拆解员 ${i}。当前目录是论文库,下面这批 PDF 在本目录。用 \`pdftotext -f 1 -l 8 '文件名' -\` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(${i})==
${PAPERS}

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。"
  nohup env ${PX} ${CODEX} --search exec -s read-only --color never "${PROMPT}" > "$OUT/reviews/deep_${i}.md" 2>&1 &
  echo "launched deep-codex ${i} (PID $!)"
  sleep 2
done
echo "=== deep-codex 全部启动, 共 ${i} 个 ==="
