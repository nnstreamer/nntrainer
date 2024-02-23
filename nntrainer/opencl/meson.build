opencl_sources = [
    'opencl_buffer.cpp',
    'opencl_command_queue_manager.cpp',
    'opencl_context_manager.cpp',
    'opencl_kernel.cpp',
    'opencl_loader.cpp',
    'opencl_program.cpp',
    'opencl_op_interface.cpp'
]

opencl_headers = [
  'third_party/cl_platform.h',
  'opencl_op_interface.h',
  'opencl_command_queue_manager.h',
  'opencl_context_manager.h',
  'opencl_kernel.h',
  'opencl_program.h'
]

foreach s : opencl_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : opencl_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
