#!/usr/bin/env python3
"""
Benchmark HTML Report Generator

This script generates HTML reports for GitHub Pages from benchmark results.
It creates a nice dashboard with charts and historical data.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class HTMLGenerator:
    """Generate HTML reports from benchmark results"""
    
    def __init__(self):
        self.template = self.get_html_template()
    
    def get_html_template(self) -> str:
        """Get the HTML template for the benchmark report"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results - {project_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f6f8fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e1e4e8;
        }}
        .header h1 {{
            color: #24292e;
            margin: 0;
        }}
        .subtitle {{
            color: #586069;
            margin-top: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #24292e;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #0366d6;
        }}
        .metric-unit {{
            color: #586069;
            font-size: 0.9em;
        }}
        .context-info {{
            background: #f1f8ff;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #c8e1ff;
            margin-bottom: 20px;
        }}
        .context-info h3 {{
            margin: 0 0 10px 0;
            color: #0366d6;
        }}
        .context-item {{
            margin: 5px 0;
            font-family: monospace;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e1e4e8;
            color: #586069;
            font-size: 0.9em;
        }}
        .timestamp {{
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ffeaa7;
            margin-bottom: 20px;
            text-align: center;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
        }}
        .no-data {{
            text-align: center;
            color: #586069;
            padding: 40px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Benchmark Results</h1>
            <p class="subtitle">Performance Metrics Dashboard</p>
        </div>
        
        <div class="timestamp">
            <strong>Last Updated:</strong> {timestamp}
        </div>
        
        <div class="context-info">
            <h3>📊 Run Context</h3>
            {context_html}
        </div>
        
        <div class="metrics-grid">
            {metrics_html}
        </div>
        
        <div class="chart-container">
            <h3>📈 Historical Trends</h3>
            <div class="no-data">
                Historical charts will be available after multiple benchmark runs.
                <br>
                <em>Consider implementing Chart.js or similar charting library for trends.</em>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated automatically by GitHub Actions</p>
            <p>Report generated on {generation_time}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_metrics_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML for metrics cards"""
        metrics_html = ""
        
        # Handle simple benchmark results
        if 'peak_memory_mb' in results:
            metrics_html += f"""
            <div class="metric-card">
                <h3>🧠 Peak Memory Usage</h3>
                <div class="metric-value">{results['peak_memory_mb']}</div>
                <div class="metric-unit">MB</div>
            </div>
            """
        
        if 'cpu_cycles' in results:
            metrics_html += f"""
            <div class="metric-card">
                <h3>⚡ CPU Cycles</h3>
                <div class="metric-value">{results['cpu_cycles']:,}</div>
                <div class="metric-unit">cycles</div>
            </div>
            """
        
        # Handle Google Benchmark results
        for name, data in results.items():
            if isinstance(data, dict) and 'time' in data:
                metrics_html += f"""
                <div class="metric-card">
                    <h3>⏱️ {name}</h3>
                    <div class="metric-value">{data['time']:.2f}</div>
                    <div class="metric-unit">ns</div>
                </div>
                """
        
        if not metrics_html:
            metrics_html = """
            <div class="metric-card">
                <h3>⚠️ No Metrics Found</h3>
                <div class="metric-value">-</div>
                <div class="metric-unit">Please check benchmark output</div>
            </div>
            """
        
        return metrics_html
    
    def generate_context_html(self, context: Dict[str, Any]) -> str:
        """Generate HTML for context information"""
        context_html = ""
        
        important_fields = {
            'commit_sha': '🔗 Commit SHA',
            'branch': '🌿 Branch',
            'workflow': '🔄 Workflow',
            'run_number': '🔢 Run Number',
            'actor': '👤 Actor',
            'event_name': '📅 Event'
        }
        
        for field, label in important_fields.items():
            value = context.get(field, 'unknown')
            if field == 'commit_sha' and value != 'unknown':
                # Make commit SHA a clickable link if we have it
                short_sha = value[:7]
                context_html += f'<div class="context-item"><strong>{label}:</strong> <code>{short_sha}</code></div>'
            else:
                context_html += f'<div class="context-item"><strong>{label}:</strong> <code>{value}</code></div>'
        
        return context_html
    
    def generate_report(self, data: Dict[str, Any], project_name: str = "Project") -> str:
        """Generate complete HTML report"""
        timestamp = data.get('timestamp', datetime.now().isoformat())
        results = data.get('results', {})
        context = data.get('context', {})
        
        metrics_html = self.generate_metrics_html(results)
        context_html = self.generate_context_html(context)
        
        html = self.template.format(
            project_name=project_name,
            timestamp=timestamp,
            context_html=context_html,
            metrics_html=metrics_html,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        
        return html


def main():
    """Main function to generate HTML report"""
    parser = argparse.ArgumentParser(description='Generate HTML report from benchmark results')
    parser.add_argument('input_file', help='Input JSON file with benchmark results')
    parser.add_argument('output_file', help='Output HTML file')
    parser.add_argument('--project-name', default='Project', help='Project name for the report')
    
    args = parser.parse_args()
    
    # Read input data
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input_file}")
        sys.exit(1)
    
    # Generate HTML
    generator = HTMLGenerator()
    html = generator.generate_report(data, args.project_name)
    
    # Write output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated: {args.output_file}")


if __name__ == '__main__':
    main()
