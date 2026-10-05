#!/bin/bash
# Stop tool_spec_check runs of one group (default opinfo): runner scripts, xargs, timeout wrappers, python workers.
# Matches on the start of the command line only, so the calling shell (whose own command text may contain the same
# words) is never selected.
#   scripts/stop_tool_runs.sh [GROUP]
G=${1:-opinfo}
pids=$(ps -eo pid=,args= | awk -v g="$G" '
  ($2 ~ /^(\/bin\/bash|bash)$/ && $3 ~ /run_tool_cases_(parallel|chunked)\.sh$/ && $4 == g) ||
  ($2 == "xargs" && $3 == "-P") ||
  ($2 == "timeout" && $4 == "/data1/tzh/envs/ka_main/bin/python" && $5 ~ /tool_spec_check\.py$/ && $7 == g) ||
  ($2 == "/data1/tzh/envs/ka_main/bin/python" && $3 ~ /tool_spec_check\.py$/ && $5 == g) {print $1}')
[ -n "$pids" ] && kill $pids 2>/dev/null
sleep 3
left=$(ps -eo pid=,args= | awk -v g="$G" '$2 == "/data1/tzh/envs/ka_main/bin/python" && $3 ~ /tool_spec_check\.py$/ && $5 == g {print $1}')
[ -n "$left" ] && kill -9 $left 2>/dev/null
echo "stopped: $(echo $pids | wc -w) processes"
