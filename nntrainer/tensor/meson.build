tensor_sources = [

  'cache_elem.cpp',
  'cache_loader.cpp',
  'cache_pool.cpp',
  'lazy_tensor.cpp',
  'manager.cpp',
  'tensor.cpp',
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
  'weight.h',
  'var_grad.h',    
  'tensor_wrap_specs.h',

]

cl_sources = [
  'cl_operations/cl_sgemv.cpp'
]

cl_headers = [
  'cl_operations/cl_interface.h'
]

subdir('cpu_backend')
nntrainer_inc += include_directories('cpu_backend')
nntrainer_inc_abs += meson.current_source_dir() / 'cpu_backend'


if get_option('enable-opencl')
  tensor_sources += cl_sources
  tensor_headers += cl_headers
endif

foreach s : tensor_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : tensor_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
