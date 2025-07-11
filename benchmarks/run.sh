#!/bin/bash

# Placeholder benchmark script that will be replaced with actual benchmark code
# This script simulates getting performance metrics as requested

set -e

echo "Running benchmark..."

# Simulate running some benchmark and getting metrics
# In real implementation, this would run actual C/C++ benchmarks
pushd benchmark_application
perf stat -e cycles /usr/bin/time -f %M ./Benchmark_ResNet 2> ../result
popd

# Extract the first number-only line.
PEAK_MEMORY_KB=`grep "^[0-9][0-9]*$" result`
PEAK_MEMORY_MB=$((PEAK_MEMORY_MB / 1000))
# Extract the number (#########) from "       ###,###,###,###    cycles".
CPU_CYCLES=`grep -o "[0-9,][0-9,]*.*cycles" result | grep -o "[0-9,]*" | sed  "s|,||g"`

# Output the metrics to stdout as specified
echo "Peak Memory (MB): $PEAK_MEMORY_MB"
echo "CPU Cycles: $CPU_CYCLES"

echo "Benchmark completed successfully"
