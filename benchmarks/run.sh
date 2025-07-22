#!/bin/bash

# Placeholder benchmark script that will be replaced with actual benchmark code
# This script simulates getting performance metrics as requested

set -e

BENCH_TARGET=""
BENCH_RESULT=""


if [ -z "$1" ]; then
  BENCH_TARGET="../build_benchmarks/benchmarks/benchmark_application/Benchmark_ResNet"
  echo "1st argument (benchmark executable) not supplied. Using the default: $BENCH_TARGET"
else
  BENCH_TARGET="$1"
fi
if [ -z "$2" ]; then
  BENCH_RESULT="benchmark_output.txt"
  echo "2nd argument (benchmark output) not supplied. Using the default: $BENCH_RESULT"
else
  BENCH_RESULT="$2"
fi
echo "Running benchmark of $BENCH_TARGET, written to $BENCH_RESULT"

# Simulate running some benchmark and getting metrics
# In real implementation, this would run actual C/C++ benchmarks
TMPF=$(mktemp)
sudo perf stat -e cycles /usr/bin/time -f %M ${BENCH_TARGET} 2> ${TMPF} || cat ${TMPF}

cat ${TMPF}

# Extract the first number-only line.
PEAK_MEMORY_KB=`grep "^[0-9][0-9]*$" ${TMPF}`
PEAK_MEMORY_MB=$((PEAK_MEMORY_KB / 1000))
# Extract the number (#########) from "       ###,###,###,###    cycles".
CPU_CYCLES=`grep -o "[0-9,][0-9,]*.*cycles" ${TMPF} | grep -o "[0-9,]*" | sed  "s|,||g"`
TIME=`grep -o "[0-9,][0-9,]*\.[0-9]* seconds time elapsed" ${TMPF} | grep -o "[0-9,][0-9,]*\.[0-9]*" | sed "s|,||g"`

# Output the metrics to stdout as specified
echo "Peak Memory (MB): $PEAK_MEMORY_MB"
echo "CPU Cycles: $CPU_CYCLES"
echo "Wall Time (s): $TIME"
echo "Peak Memory (MB): $PEAK_MEMORY_MB" > ${BENCH_RESULT}
echo "CPU Cycles: $CPU_CYCLES" >> ${BENCH_RESULT}
echo "Wall Time (s): $TIME" >> ${BENCH_RESULT}

echo "Benchmark completed successfully"
