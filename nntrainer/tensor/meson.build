tensor_sources = [
  'blas_interface.cpp',
  'lazy_tensor.cpp',
  'tensor.cpp',
  'tensor_dim.cpp',
  'var_grad.cpp',
  'weight.cpp'
]

tensor_headers = [
  'tensor.h',
  'tensor_dim.h',
  'weight.h',
  'var_grad.h'
]

foreach s : tensor_sources
  nntrainer_sources += join_paths(meson.current_source_dir(), s)
endforeach

foreach h : tensor_headers
  nntrainer_headers += join_paths(meson.current_source_dir(), h)
endforeach
