cl_layer_sources = [
  'fc_layer_cl.cpp',
  'blas_kernels.cpp',
]

if get_option('enable-fp16')
  cl_layer_sources += 'blas_kernels_fp16.cpp'
endif

foreach s : cl_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
