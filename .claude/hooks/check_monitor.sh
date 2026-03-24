#!/bin/bash
# Hook: 阻止连续 sleep(>=180s) 而不更新 monitor.md
# 短 sleep（<180s）不受限制，长 sleep 前必须先写 monitor.md

INPUT=$(cat)

COMMAND=$(python3 -c "
import json
try:
    data = json.loads('''$INPUT''')
    print(data.get('tool_input', {}).get('command', ''))
except:
    pass
" 2>/dev/null)

# 只检查 sleep 命令
if echo "$COMMAND" | grep -qE '^sleep '; then
  # 提取 sleep 秒数
  SLEEP_SEC=$(echo "$COMMAND" | grep -oP 'sleep\s+\K\d+')

  # 只对 >= 180 秒的 sleep 生效
  if [ -n "$SLEEP_SEC" ] && [ "$SLEEP_SEC" -ge 179 ]; then
    MARKER="/tmp/.claude_last_sleep"

    if [ -f "$MARKER" ]; then
      LAST_SLEEP=$(cat "$MARKER")

      # 检查是否有任何 monitor.md 在上次 sleep 之后被修改过
      MONITOR_UPDATED=false
      for f in experiments/exp*/monitor.md; do
        if [ -f "$f" ]; then
          MOD_TIME=$(stat -c %Y "$f" 2>/dev/null)
          if [ -n "$MOD_TIME" ] && [ "$MOD_TIME" -gt "$LAST_SLEEP" ]; then
            MONITOR_UPDATED=true
            break
          fi
        fi
      done

      if [ "$MONITOR_UPDATED" = false ]; then
        cat <<'ENDJSON'
{"decision":"block","reason":"You have NOT updated any monitor.md since your last sleep. Check training logs and write observations to experiments/exp{NNN}/monitor.md first."}
ENDJSON
        exit 2
      fi
    fi

    # 记录本次 sleep 的时间戳（仅在放行时记录）
    date +%s > "$MARKER"
  fi
fi

exit 0
