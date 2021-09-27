tensor_sources = [
  'blas_interface.cpp',
  'lazy_tensor.cpp',
  'manager.cpp',
  'tensor.cpp',
  'tensor_dim.cpp',
  'var_grad.cpp',
  'weight.cpp',
  'basic_planner.cpp',
  'memory_pool.cpp'
]

tensor_headers = [
  'tensor.h',
  'tensor_wrap_specs.h'
]

foreach s : tensor_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : tensor_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
