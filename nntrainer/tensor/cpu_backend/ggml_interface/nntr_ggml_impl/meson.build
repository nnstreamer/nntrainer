ggml_impl_interface_headers = [
    'nntr_ggml_impl.h',
    'nntr_ggml_impl_common.h',
    'nntr_ggml_impl_utils.h',
]

ggml_impl_interface_sources = [
  'nntr_ggml_impl.cpp',
  'nntr_ggml_impl_quant.cpp',
  'nntr_ggml_impl_fp16_fp32.cpp',
]

foreach s : ggml_impl_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_impl_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
