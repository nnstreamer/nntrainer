ggml_interface_headers = [
    'ggml_interface.h',
]
ggml_interface_sources = [ ]

if get_option('ggml-thread-backend') == 'bsthread'
  ggml_interface_sources += 'ggml_interface_bs_threadpool.cpp'
elif get_option('ggml-thread-backend') == 'openmp'
  ggml_interface_sources += 'ggml_interface_omp.cpp'
elif get_option('ggml-thread-backend') == 'spthread'
  ggml_interface_sources += 'ggml_interface_sp_threadpool.cpp'
else
  message('Unknown ggml-thread-backend meson option! Use openMP as a default.')
  ggml_interface_sources += 'ggml_interface_omp.cpp'
endif

foreach s : ggml_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
