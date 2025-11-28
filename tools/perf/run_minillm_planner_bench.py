#!/usr/bin/env python3
"""Compile and run a tiny memory-planner benchmark for a small LLM graph."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import textwrap


CPP_SOURCE = textwrap.dedent(
    """
    #include <chrono>
    #include <iostream>
    #include <utility>
    #include <vector>

    #include <optimized_v1_planner.h>
    #include <optimized_v3_planner.h>

    int main() {
      using namespace nntrainer;
      const std::vector<size_t> memory_size = {2048, 1024, 768, 512, 768, 512};
      const std::vector<std::pair<unsigned int, unsigned int>> memory_validity = {
        {0, 2}, {0, 8}, {2, 4}, {3, 5}, {4, 6}, {6, 8}};
      const size_t n = memory_size.size();

      OptimizedV1Planner v1;
      OptimizedV3Planner v3;
      std::vector<bool> is_wgrad(n, false);

      auto run_once = [&](auto &planner) {
        std::vector<size_t> local_offsets(n);
        std::vector<bool> local_wgrad = is_wgrad;
        return planner.planLayout(memory_size, memory_validity, local_offsets,
                                  local_wgrad, 0);
      };

      auto benchmark = [&](auto &planner) {
        constexpr size_t iterations = 20000;
        auto begin = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
          std::vector<size_t> local_offsets(n);
          std::vector<bool> local_wgrad = is_wgrad;
          (void)planner.planLayout(memory_size, memory_validity, local_offsets,
                                   local_wgrad, 0);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto total = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - begin);
        return total.count() / iterations;
      };

      const size_t peak_v1 = run_once(v1);
      const size_t peak_v3 = run_once(v3);
      const double avg_time_v1 = benchmark(v1);
      const double avg_time_v3 = benchmark(v3);

      std::cout << "peak_v1=" << peak_v1 << " bytes" << std::endl;
      std::cout << "peak_v3=" << peak_v3 << " bytes" << std::endl;
      std::cout << "avg_time_v1=" << avg_time_v1 << " us" << std::endl;
      std::cout << "avg_time_v3=" << avg_time_v3 << " us" << std::endl;
      return 0;
    }
    """
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cpp_path = tmpdir_path / "minillm_planner_bench.cpp"
        bin_path = tmpdir_path / "minillm_planner_bench"
        cpp_path.write_text(CPP_SOURCE)

        include_args = [
            f"-I{repo_root}",
            f"-I{repo_root / 'nntrainer'}",
            f"-I{repo_root / 'nntrainer' / 'tensor'}",
        ]
        compile_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            *include_args,
            str(cpp_path),
            str(repo_root / "nntrainer" / "tensor" / "optimized_v1_planner.cpp"),
            str(repo_root / "nntrainer" / "tensor" / "optimized_v3_planner.cpp"),
            "-o",
            str(bin_path),
        ]
        subprocess.run(compile_cmd, check=True, cwd=repo_root)
        result = subprocess.run([str(bin_path)], check=True, cwd=repo_root, capture_output=True, text=True)
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
