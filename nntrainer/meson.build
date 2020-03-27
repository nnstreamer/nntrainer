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
  'src/layers.cpp',
  'src/databuffer.cpp'
]

nntrainer_headers = [
  'include/neuralnet.h',
  'include/tensor.h',
  'include/layers.h',
  'include/databuffer.h'
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
