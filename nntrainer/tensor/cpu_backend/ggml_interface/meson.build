ggml_interface_headers = [
    'ggml_interface.h',
    'nntr_ggml_impl.h',
]

ggml_interface_sources = [
  'nntr_ggml_impl.cpp',
  # 'ggml_interface.cpp'
]

if get_option('ggml-thread-backend') == 'bstp'
  ggml_interface_sources += 'ggml_interface_bs_threadpool.cpp'
elif get_option('ggml-thread-backend') == 'omp'
  ggml_interface_sources += 'ggml_interface_omp.cpp'
elif get_option('ggml-thread-backend') == 'mixed'
  ggml_interface_sources += 'ggml_interface_mixed.cpp'
else
  message('Unknown ggml-thread-backend meson option! Use omp as a default.')
  ggml_interface_sources += 'ggml_interface_omp.cpp'
endif

if get_option('enable-fp16')
  ggml_interface_sources += 'ggml_interface_fp16.cpp'
endif

foreach s : ggml_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
