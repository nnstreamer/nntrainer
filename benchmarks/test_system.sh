#!/bin/bash

# Test script for the benchmarking system
# This script validates that all components work together correctly

set -e

echo "🧪 Testing Benchmarking System"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

test_passed() {
    echo -e "${GREEN}✓ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_failed() {
    echo -e "${RED}✗ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

test_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Change to benchmarks directory
cd "$(dirname "$0")"

test_info "Testing from directory: $(pwd)"

# Test 1: Check if run.sh exists and is executable
echo
echo "Test 1: Checking run.sh script..."
if [ -f "run.sh" ] && [ -x "run.sh" ]; then
    test_passed "run.sh exists and is executable"
else
    test_failed "run.sh is missing or not executable"
fi

# Test 2: Check if Python scripts exist and are executable
echo
echo "Test 2: Checking Python scripts..."
for script in "parse_results.py" "generate_html.py"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        test_passed "$script exists and is executable"
    else
        test_failed "$script is missing or not executable"
    fi
done

# Test 3: Run the benchmark script
echo
echo "Test 3: Running benchmark script..."
if ./run.sh > test_output.txt 2>&1; then
    test_passed "run.sh executed successfully"
    
    # Check if output contains expected patterns
    if grep -q "Peak Memory (MB):" test_output.txt && grep -q "CPU Cycles:" test_output.txt; then
        test_passed "Benchmark output contains expected metrics"
    else
        test_failed "Benchmark output missing expected metrics"
        echo "Output was:"
        cat test_output.txt
    fi
else
    test_failed "run.sh failed to execute"
fi

# Test 4: Parse benchmark results
echo
echo "Test 4: Parsing benchmark results..."
if python3 parse_results.py test_output.txt > test_results.json 2>&1; then
    test_passed "parse_results.py executed successfully"
    
    # Check if JSON is valid
    if python3 -c "import json; json.load(open('test_results.json'))" 2>/dev/null; then
        test_passed "Generated valid JSON output"
        
        # Check if JSON contains expected fields
        if python3 -c "import json; data=json.load(open('test_results.json')); assert 'results' in data and 'context' in data and 'timestamp' in data" 2>/dev/null; then
            test_passed "JSON contains expected structure"
        else
            test_failed "JSON missing expected structure"
        fi
    else
        test_failed "Generated invalid JSON"
    fi
else
    test_failed "parse_results.py failed to execute"
fi

# Test 5: Generate HTML report
echo
echo "Test 5: Generating HTML report..."
mkdir -p test_output
if python3 generate_html.py test_results.json test_output/test_report.html --project-name "Test Project" 2>&1; then
    test_passed "generate_html.py executed successfully"
    
    # Check if HTML file was created
    if [ -f "test_output/test_report.html" ]; then
        test_passed "HTML report file created"
        
        # Check if HTML contains expected content
        if grep -q "Benchmark Results" test_output/test_report.html && grep -q "Peak Memory Usage" test_output/test_report.html; then
            test_passed "HTML report contains expected content"
        else
            test_failed "HTML report missing expected content"
        fi
    else
        test_failed "HTML report file not created"
    fi
else
    test_failed "generate_html.py failed to execute"
fi

# Test 6: Test with different benchmark types
echo
echo "Test 6: Testing Google Benchmark parsing..."
cat > test_google_benchmark.json << 'EOF'
{
  "benchmarks": [
    {
      "name": "TestBenchmark",
      "real_time": 1234.5,
      "cpu_time": 1200.0,
      "iterations": 1000
    }
  ]
}
EOF

if python3 parse_results.py test_google_benchmark.json google_benchmark > test_google_results.json 2>&1; then
    test_passed "Google Benchmark parsing works"
else
    test_failed "Google Benchmark parsing failed"
fi

# Test 7: Check Python dependencies
echo
echo "Test 7: Checking Python dependencies..."
python3 -c "import json, sys, re, os, datetime, argparse" 2>/dev/null && test_passed "All Python dependencies available" || test_failed "Missing Python dependencies"

# Test 8: Validate file structure
echo
echo "Test 8: Validating file structure..."
expected_files=("run.sh" "parse_results.py" "generate_html.py" "README.md")
for file in "${expected_files[@]}"; do
    if [ -f "$file" ]; then
        test_passed "$file exists"
    else
        test_failed "$file missing"
    fi
done

# Clean up test files
echo
echo "Cleaning up test files..."
rm -f test_output.txt test_results.json test_google_benchmark.json test_google_results.json
rm -rf test_output/
test_info "Test files cleaned up"

# Summary
echo
echo "=============================="
echo "Test Summary:"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✨${NC}"
    echo "The benchmarking system is ready to use."
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the output above.${NC}"
    exit 1
fi
