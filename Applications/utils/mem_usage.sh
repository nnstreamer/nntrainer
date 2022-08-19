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
peak=0
mem_peak=0
log="\nPID - $process_id :"
outlog=""
count=0
while true; do
    if [ -d "/proc/$process_id" ]; then
	output=`grep VmHWM /proc/$process_id/status`
	read -ra mem_str <<< "$output"
	count=$((count + 1))
	mem_usage=$((${mem_str[1]}))
	cpu_output=""
	n=`nproc`
	i=0
	while [ "$i" -lt $n ]
	do
	    cpu_freq=`cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq`
	    cpu_output="$cpu_output $cpu_freq"
	    i=$(expr $i + 1)
	done
	outlog="$outlog$count $mem_usage kB $cpu_output\n"
	let mem_peak='mem_usage > mem_peak ? mem_usage : mem_peak'
	sleep 1s # configure the sleep time if the program does not run long enough
    else
	break
    fi
done

if [ -d "/proc/$process_id" ]; then
    kill -9 $process_id
fi

log="$log: $mem_peak kB"
log="$log\n-------------------------------------\n"
log="$log$outlog"

echo -e "$log"



