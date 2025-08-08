benchmark_ggml_deps = [
  nntrainer_dep,
  nntrainer_testutil_dep,
]

benchmark_ggml = executable(
  'benchmark_ggml_threadpool',
  'benchmark_ggml_threadpool.cpp',
  dependencies: benchmark_ggml_deps,
  include_directories: nntrainer_test_inc,
  install: false
)