tensor_sources = [
  'blas_interface.cpp',
  'cache_elem.cpp',
  'cache_loader.cpp',
  'cache_pool.cpp',
  'lazy_tensor.cpp',
  'manager.cpp',
  'tensor.cpp',
  'tensor_v2.cpp',
  'float_tensor.cpp',
  'tensor_dim.cpp',
  'var_grad.cpp',
  'weight.cpp',
  'basic_planner.cpp',
  'memory_pool.cpp',
  'swap_device.cpp',
  'tensor_pool.cpp',
  'optimized_v1_planner.cpp',
  'optimized_v2_planner.cpp',
  'optimized_v3_planner.cpp',
  'task_executor.cpp',
]

tensor_headers = [
  'memory_data.h',
  'tensor.h',
  'tensor_v2.h',
  'tensor_base.h',
  'float_tensor.h',
  'half_tensor.h',
  'weight.h',
  'var_grad.h',    
  'tensor_wrap_specs.h',
  'blas_interface.h'
]

arch = target_machine.cpu_family()

if get_option('platform') == 'android' or arch == 'aarch64'
  tensor_sources += 'blas_neon.cpp'
endif

if get_option('enable-fp16')
  tensor_sources += 'half_tensor.cpp'
endif

foreach s : tensor_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : tensor_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
