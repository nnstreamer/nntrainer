nntrainer_inc = include_directories('./include')

# Dependencies
nntrainer_base_deps=[
  blas_dep,
  iniparser_dep,
  libm_dep,
  libdl_dep,
  thread_dep
]

if build_platform == 'tizen'
  nntrainer_base_deps += dependency('dlog')
endif

nntrainer_sources = [
  'src/neuralnet.cpp',
  'src/tensor.cpp',
  'src/layer.cpp',
  'src/input_layer.cpp',
  'src/fc_layer.cpp',
  'src/bn_layer.cpp',
  'src/databuffer.cpp',
  'src/databuffer_file.cpp',
  'src/databuffer_func.cpp',
  'src/nntrainer_logger.cpp',
  'src/optimizer.cpp',
  'src/util_func.cpp',
  'src/parse_util.cpp',
  'src/tensor_dim.cpp'
]

nntrainer_headers = [
  'include/neuralnet.h',
  'include/tensor.h',
  'include/layer.h',
  'include/input_layer.h',
  'include/fc_layer.h',
  'include/bn_layer.h',
  'include/databuffer.h',
  'include/databuffer_file.h',
  'include/databuffer_func.h',
  'include/nntrainer_log.h',
  'include/nntrainer_logger.h',
  'include/optimizer.h',
  'include/util_func.h',
  'include/parse_util.h',
  'include/tensor_dim.h'
]


# Build libraries
nntrainer_shared = shared_library('nntrainer',
  nntrainer_sources,
  dependencies: nntrainer_base_deps,
  include_directories: nntrainer_inc,
  install: true,
  install_dir: nntrainer_libdir
)

nntrainer_static = static_library('nntrainer',
  nntrainer_sources,
  dependencies: nntrainer_base_deps,
  include_directories: nntrainer_inc,
  install: true,
  install_dir: nntrainer_libdir
)

nntrainer_lib = nntrainer_shared
if get_option('default_library') == 'static'
  nntrainer_lib = nntrainer_static
endif

nntrainer_dep = declare_dependency(link_with: nntrainer_lib,
  dependencies: nntrainer_base_deps,
  include_directories: nntrainer_inc
)

# Install headers
install_headers(nntrainer_headers,
  subdir: 'nntrainer'
)
