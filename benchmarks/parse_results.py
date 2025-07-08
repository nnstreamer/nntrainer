#!/usr/bin/env python3
"""
Benchmark Results Parser

This script parses benchmark output and generates structured data for GitHub Pages.
It's designed to be easily extensible for future benchmark types.
"""

import json
import sys
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class BenchmarkParser:
    """Parse and process benchmark results"""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().isoformat()
    
    def parse_simple_output(self, output: str) -> Dict[str, Any]:
        """
        Parse simple benchmark output format:
        Peak Memory (MB): 123
        CPU Cycles: 456789
        """
        results = {}
        
        # Parse peak memory
        memory_match = re.search(r'Peak Memory \(MB\):\s*(\d+)', output)
        if memory_match:
            results['peak_memory_mb'] = int(memory_match.group(1))
        
        # Parse CPU cycles
        cycles_match = re.search(r'CPU Cycles:\s*(\d+)', output)
        if cycles_match:
            results['cpu_cycles'] = int(cycles_match.group(1))
        
        return results
    
    def parse_google_benchmark_output(self, output: str) -> Dict[str, Any]:
        """
        Parse Google Benchmark JSON output format
        This can be extended when actual benchmark code is ready
        """
        results = {}
        
        # Try to parse as JSON first
        try:
            benchmark_data = json.loads(output)
            if 'benchmarks' in benchmark_data:
                for benchmark in benchmark_data['benchmarks']:
                    name = benchmark.get('name', 'unknown')
                    results[name] = {
                        'time': benchmark.get('real_time', 0),
                        'cpu_time': benchmark.get('cpu_time', 0),
                        'iterations': benchmark.get('iterations', 0)
                    }
        except json.JSONDecodeError:
            # Fallback to simple parsing
            pass
        
        return results
    
    def parse_output(self, output: str, benchmark_type: str = 'simple') -> Dict[str, Any]:
        """Parse benchmark output based on type"""
        if benchmark_type == 'simple':
            return self.parse_simple_output(output)
        elif benchmark_type == 'google_benchmark':
            return self.parse_google_benchmark_output(output)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    def generate_result_summary(self, parsed_results: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete result summary"""
        summary = {
            'timestamp': self.timestamp,
            'context': context,
            'results': parsed_results,
            'metadata': {
                'parser_version': '1.0',
                'format_version': '1.0'
            }
        }
        return summary


def main():
    """Main function to parse benchmark results"""
    if len(sys.argv) < 2:
        print("Usage: python3 parse_results.py <benchmark_output_file> [benchmark_type]")
        sys.exit(1)
    
    output_file = sys.argv[1]
    benchmark_type = sys.argv[2] if len(sys.argv) > 2 else 'simple'
    
    # Read benchmark output
    try:
        with open(output_file, 'r') as f:
            output = f.read()
    except FileNotFoundError:
        print(f"Error: File {output_file} not found")
        sys.exit(1)
    
    # Parse results
    parser = BenchmarkParser()
    parsed_results = parser.parse_output(output, benchmark_type)
    
    # Get context information
    context = {
        'commit_sha': os.environ.get('GITHUB_SHA', 'unknown'),
        'branch': os.environ.get('GITHUB_REF_NAME', 'unknown'),
        'workflow': os.environ.get('GITHUB_WORKFLOW', 'unknown'),
        'run_number': os.environ.get('GITHUB_RUN_NUMBER', 'unknown'),
        'actor': os.environ.get('GITHUB_ACTOR', 'unknown'),
        'event_name': os.environ.get('GITHUB_EVENT_NAME', 'unknown')
    }
    
    # Generate summary
    summary = parser.generate_result_summary(parsed_results, context)
    
    # Output JSON
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
