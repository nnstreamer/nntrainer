ggml_interface_headers = [
  'ggml_interface.h',
]
ggml_interface_sources = [
  'ggml_interface.cpp',
]

foreach s : ggml_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
