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
  'opencl_op_interface.hpp'
]

foreach s : opencl_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : opencl_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach