#!/usr/bin/env bash
##
## @file mem_usage.sh
## @author Parichay Kapoor <pk.kapoor@samsung.com>
## @date 13 September 2020
## @brief peak memory usage for the program
##
## @note the program must be running for at least 30 sec to measure the memory
## usage. If the program finishes earlier, reduce the sleep time

"$@" &
process_id=$!

if [ -d "/proc/$process_id" ]; then
  sleep 30s # configure the sleep time if the program does not run long enough
  output=`grep VmHWM /proc/$process_id/status`
  IFS=' ' read -ra mem_str <<< "$output"
  mem_usage=$((${mem_str[1]}))
fi
echo "Mem Usage: $mem_usage"

kill -9 $process_id

