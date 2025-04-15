ggml_gemm_wo_ggml_tensor_headers = [
  'ggml_avx_macro.h',
  'ggml_cpu_impl.h',
  'ggml-common.h',
  'nntr_ggml_essential.h',
  'ggml_interface.h',
]

ggml_gemm_wo_ggml_tensor_sources = [
  'ggml_cpu_impl.cpp',
  'ggml_interface.cpp',
]

foreach s : ggml_gemm_wo_ggml_tensor_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_gemm_wo_ggml_tensor_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
