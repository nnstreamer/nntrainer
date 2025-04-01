ggml_gemm_headers = [
  'ggml_avx_macro.h',
  'ggml_cpu_gemm.h',
  'ggml-alloc.h',
  'ggml-backend-impl.h',
  'ggml-backend.h',
  'ggml-common.h',
  'ggml-cpu-impl.h',
  'ggml-cpu-traits.h',
  'ggml-cpu.h',
  'ggml-quants.h',
  'common.h',
  'binary-ops.h',
  'ggml-threading.h',
  'ggml.h',
  'unary-ops.h'
]
ggml_gemm_sources = [
    'ggml_cpu_gemm.cpp',
    'ggml-cpu-traits.cpp',
    'ggml-cpu.cpp',
    'ggml-quants.c',
    'ggml-threading.cpp',
    'ggml.c',
    'unary-ops.cpp',
    'binary-ops.cpp',
    'ggml-quants.c',
    'ggml-cpu.c',
    'ggml-cpu-quants.c',
    'ggml-cpu.c',
]

foreach s : ggml_gemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_gemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
