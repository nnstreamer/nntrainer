ggml_interface_headers = [
    'ggml_interface.h',
]
ggml_interface_sources = [ ]

if get_option('K-quant_multithreading_strategy') == 'bs'
  ggml_interface_sources += 'ggml_interface_bs_threadpool.cpp'
elif get_option('K-quant_multithreading_strategy') == 'omp'
  ggml_interface_sources += 'ggml_interface_omp.cpp'
else
  ggml_interface_sources += 'ggml_interface_omp.cpp'
endif

foreach s : ggml_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
