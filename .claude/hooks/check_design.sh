#!/bin/bash
# Hook: 阻止不满足前置条件的实验启动训练
# 检查项: 1) design.md 存在  2) claude_review 审查通过
# 触发: PreToolUse on Bash commands containing train.py

INPUT=$(cat)

# 用 python 解析 JSON (stdin 方式，避免引号问题)
COMMAND=$(echo "$INPUT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('command', ''))
except:
    pass
" 2>/dev/null)

# 只检查包含 train.py 的命令
if echo "$COMMAND" | grep -qE 'train\.py'; then
  # 从命令中提取实验编号 (如 exp153, exp030a 等)
  EXP_ID=$(echo "$COMMAND" | grep -oP 'exp\d{3}' | head -1)

  if [ -n "$EXP_ID" ]; then
    # 找到实验目录
    EXP_DIR=""
    for dir in experiments/${EXP_ID}*/; do
      if [ -d "$dir" ]; then
        EXP_DIR="$dir"
        break
      fi
    done

    # 检查 1: design.md 必须存在
    if [ -z "$EXP_DIR" ] || [ ! -f "${EXP_DIR}design.md" ]; then
      echo "{\"decision\":\"block\",\"reason\":\"experiments/${EXP_ID}*/design.md does not exist. You must create the experiment design document before starting training.\"}"
      exit 2
    fi

    # 检查 2: claude_review 必须存在且审查通过
    REVIEW_PASSED=false
    # 检查所有版本的审查文件，找最新的
    LATEST_REVIEW=""
    for review in "${EXP_DIR}"claude_review*.md; do
      if [ -f "$review" ]; then
        LATEST_REVIEW="$review"
      fi
    done

    if [ -z "$LATEST_REVIEW" ]; then
      echo "{\"decision\":\"block\",\"reason\":\"No claude_review.md found in ${EXP_DIR}. You must run Claude Broad Review before starting training.\"}"
      exit 2
    fi

    # 检查审查文件中是否包含通过标记
    if grep -qiE '(审查通过|REVIEW.?PASS|can start training|可以开始训练)' "$LATEST_REVIEW"; then
      REVIEW_PASSED=true
    fi

    if [ "$REVIEW_PASSED" = false ]; then
      echo "{\"decision\":\"block\",\"reason\":\"Review in ${LATEST_REVIEW} has NOT passed. Fix all issues, then re-run Claude Broad Review until it passes. The review file must contain 审查通过 to indicate approval.\"}"
      exit 2
    fi
  fi
fi

exit 0
