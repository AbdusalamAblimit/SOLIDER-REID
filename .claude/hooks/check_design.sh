#!/bin/bash
# Hook: 阻止不满足前置条件的实验启动训练
# 检查项 (2026-06-26 起 纯 codex 三审制, 省 claude token, 不再用 Opus Agent):
#   1) design.md 存在
#   2) codex_review 审查通过 (verdict approve) 且 >=50 行 (三轮全量审查应详细, 防假审查)
# 触发: PreToolUse on Bash commands containing train.py (含 pretrain.py 子串, continued-pretrain 也该审)

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" 2>/dev/null || true

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('command', ''))
except:
    pass
" 2>/dev/null)

# 只拦"运行训练脚本"命令 (含 train.py/pretrain.py); 排除审查/编译/读取命令 (codex/py_compile/grep/wc 含子串但非训练)
if echo "$COMMAND" | grep -qE 'train\.py' && ! echo "$COMMAND" | grep -qE 'codex|py_compile|[[:space:]]wc[[:space:]]|--color never'; then
  EXP_ID=$(echo "$COMMAND" | grep -oE 'exp[0-9]{3}' | head -1)

  if [ -n "$EXP_ID" ]; then
    EXP_DIR=""
    for dir in experiments/${EXP_ID}*/; do
      if [ -d "$dir" ]; then
        EXP_DIR="$dir"
        break
      fi
    done

    # 检查 1: design.md 必须存在
    if [ -z "$EXP_DIR" ] || [ ! -f "${EXP_DIR}design.md" ]; then
      echo "{\"decision\":\"block\",\"reason\":\"experiments/${EXP_ID}*/design.md does not exist. Create the experiment design document before training.\"}"
      exit 2
    fi

    # 检查 2: codex 三审 — codex_review*.md 必须存在、verdict approve、>=50 行 (三轮全量审查)
    LATEST_CODEX_REVIEW=""
    for codex_review in "${EXP_DIR}"codex_review*.md; do
      if [ -f "$codex_review" ]; then
        LATEST_CODEX_REVIEW="$codex_review"
      fi
    done

    if [ -z "$LATEST_CODEX_REVIEW" ]; then
      echo "{\"decision\":\"block\",\"reason\":\"No codex_review.md in ${EXP_DIR}. 纯 codex 三审制: run 'codex --search exec -s read-only' 三轮全量审查 (相同范围), 保存到 codex_review.md, verdict 须 approve.\"}"
      exit 2
    fi

    # verdict 必须 approve
    if ! grep -qiE '(verdict.*approve|CODEX.?PASS|codex 审查通过)' "$LATEST_CODEX_REVIEW"; then
      echo "{\"decision\":\"block\",\"reason\":\"Codex review in ${LATEST_CODEX_REVIEW} 未 approve. 修完所有 findings 后重跑 codex 直到三审 verdict 全 approve.\"}"
      exit 2
    fi

    # >=50 行: 保证记录了三轮全量审查 (防自己写假审查跳过 codex)
    REVIEW_LINES=$(wc -l < "$LATEST_CODEX_REVIEW")
    if [ "$REVIEW_LINES" -lt 50 ]; then
      echo "{\"decision\":\"block\",\"reason\":\"codex_review (${REVIEW_LINES} 行) 太短. 纯 codex 三审应记录 3 轮全量审查 (>=50 行). 别写假审查跳过 codex.\"}"
      exit 2
    fi
  fi
fi

exit 0
