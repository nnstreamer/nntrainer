# Benchmarking System

This directory contains the automated benchmarking system for the NNTrainer project. The system is designed to run performance benchmarks regularly and publish results to GitHub Pages.

## 🚀 Features

- **Automated Benchmarking**: Runs benchmarks on schedule and pull requests
- **GitHub Pages Publishing**: Publishes results to `gh-pages` branch at `/test-results/`
- **Pull Request Integration**: Comments benchmark results on pull requests
- **Extensible Architecture**: Easy to add new benchmark types and metrics
- **Beautiful Reports**: Generates HTML reports with modern UI

## 📁 Directory Structure

```
benchmarks/
├── README.md                    # This file
├── run.sh                       # Placeholder benchmark script
├── parse_results.py             # Results parser (extensible)
├── generate_html.py             # HTML report generator
├── benchmark_application/       # Existing benchmark applications
│   ├── benchmark_resnet.cpp     # ResNet benchmark
│   └── meson.build
├── fake_data_gen/              # Fake data generators
└── meson.build                 # Build configuration
```

## 🔄 How It Works

### 1. GitHub Actions Workflow

The main workflow (`.github/workflows/benchmark_and_publish.yml`) handles:

- **Scheduled Runs**: Daily at 2 AM UTC
- **Pull Request Runs**: On PRs affecting relevant files
- **Manual Triggers**: Via GitHub Actions UI

### 2. Benchmark Execution

- **Simple Benchmarks**: Via `run.sh` script (placeholder)
- Please replace or add workload later.

### 3. Results Processing

1. **Parse Results**: `parse_results.py` extracts metrics from benchmark output
2. **Generate HTML**: `generate_html.py` creates beautiful HTML reports
3. **Publish**: Results are published to GitHub Pages

### 4. Output Locations

- **GitHub Pages**: `https://username.github.io/repository/test-results/`
- **Raw Data**: Available as JSON at `/test-results/benchmark_results.json`
- **PR Comments**: Results are commented on pull requests

## 🛠️ Extending the System

### Adding New Benchmark Types

To add a new benchmark type, modify `parse_results.py`:

```python
def parse_custom_benchmark_output(self, output: str) -> Dict[str, Any]:
    """Parse custom benchmark output format"""
    results = {}
    
    # Add your parsing logic here
    # Example: Extract latency, throughput, etc.
    
    return results
```

Then update the `parse_output` method to handle your new type:

```python
def parse_output(self, output: str, benchmark_type: str = 'simple') -> Dict[str, Any]:
    if benchmark_type == 'custom':
        return self.parse_custom_benchmark_output(output)
    # ... existing code
```

### Adding New Metrics

To display new metrics in HTML reports, modify `generate_html.py`:

```python
def generate_metrics_html(self, results: Dict[str, Any]) -> str:
    # Add new metric cards
    if 'your_metric' in results:
        metrics_html += f"""
        <div class="metric-card">
            <h3>📊 Your Metric</h3>
            <div class="metric-value">{results['your_metric']}</div>
            <div class="metric-unit">your_unit</div>
        </div>
        """
```

### Creating New Benchmark Scripts

1. Create your benchmark script in the `benchmarks/` directory
2. Make it executable: `chmod +x your_benchmark.sh`
3. Update the workflow to call your script:

```yaml
- name: Run your benchmark
  run: |
    cd benchmarks
    ./your_benchmark.sh > your_output.txt 2>&1
```

## 📊 Current Metrics

The system currently tracks:

- **Peak Memory Usage**: Maximum memory consumption in MB
- **CPU Cycles**: Total CPU cycles consumed
- **Execution Time**: For Google Benchmark integration

## 🔧 Configuration

### Environment Variables

- `PROJECT_NAME`: Name displayed in HTML reports (default: "NNTrainer")

### Workflow Triggers

Edit `.github/workflows/benchmark_and_publish.yml` to change:

- **Schedule**: Modify the cron expression
- **PR Triggers**: Change file paths that trigger benchmarks
- **Build Configuration**: Adjust meson build options

### Build Configuration for Benchmarks

The system uses optimized builds for accurate performance measurement:
- `--buildtype=release`: Enables compiler optimizations (-O3)
- `-Denable-debug=false`: Disables debug symbols and assertions
- This ensures benchmark results reflect actual performance

## 📈 GitHub Pages Setup

To enable GitHub Pages publishing:

1. Go to repository **Settings** → **Pages**
2. Set **Source** to "GitHub Actions"
3. The workflow will automatically publish to `gh-pages` branch

## 🚧 Placeholder Implementation

The current `run.sh` script is a placeholder that generates random values for demonstration. Replace it with your actual benchmark implementation:

```bash
#!/bin/bash
# Your actual benchmark implementation
./your_actual_benchmark
echo "Peak Memory (MB): $(get_peak_memory)"
echo "CPU Cycles: $(get_cpu_cycles)"
```

## 📝 Expected Output Format

Your benchmark script should output metrics in this format:

```
Peak Memory (MB): 123
CPU Cycles: 456789
```

For Google Benchmark integration, output JSON format:

```json
{
  "benchmarks": [
    {
      "name": "BenchmarkName",
      "real_time": 1234.5,
      "cpu_time": 1200.0,
      "iterations": 1000
    }
  ]
}
```

## 🎨 Customizing HTML Reports

The HTML template in `generate_html.py` can be customized:

- **Styling**: Modify the CSS section
- **Layout**: Change the HTML structure
- **Charts**: Add Chart.js or similar libraries for visualizations

## 🔍 Debugging

To debug the benchmarking system:

1. **Check workflow logs**: GitHub Actions → Workflow runs
2. **Download artifacts**: Benchmark results are saved as artifacts
3. **Test locally**: Run scripts locally to verify output format
4. **Enable debug mode**: Set `ACTIONS_STEP_DEBUG=true` in repository secrets

## 🤝 Contributing

When adding new benchmarks:

1. Follow the existing patterns in `parse_results.py`
2. Add comprehensive documentation
3. Test both locally and in the GitHub Actions environment
4. Update this README with your changes

## 📋 Requirements

- Python 3.10+
- Ubuntu 22.04+ (for GitHub Actions)
- All system dependencies are installed automatically by the workflow

## 🔄 Migration from Existing Benchmarks

The system is designed to coexist with existing benchmark workflows. The current `ubuntu_benchmarks.yml` workflow can run in parallel until migration is complete.

## 🎯 Future Enhancements

- **Historical Trending**: Track performance over time
- **Regression Detection**: Automatically detect performance regressions
- **Comparison Views**: Compare performance across branches/commits
- **Integration with External Tools**: Support for additional benchmark frameworks
- **Real-time Monitoring**: Integration with monitoring dashboards

---

This system provides a solid foundation for automated performance benchmarking and can be easily extended as your benchmarking needs evolve.
