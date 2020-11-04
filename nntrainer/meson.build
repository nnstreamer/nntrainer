nntrainer_inc = [
  include_directories('.'),
  include_directories('../api', '../api/ccapi/include')
]

nntrainer_sources = []
nntrainer_headers = []

# pc file is not present for 'ml-api-common' yet
if cxx.has_header('nnstreamer/ml-api-common.h', required: false)
  nntrainer_inc += include_directories ('/usr/include/nnstreamer')
else
  nntrainer_inc += include_directories ('../api/capi/include/platform')
endif

# Dependencies
nntrainer_base_deps=[
  openmp_dep,
  blas_dep,
  iniparser_dep,
  libm_dep,
  libdl_dep,
  thread_dep
]

if build_platform == 'tizen'
  nntrainer_base_deps += dependency('dlog')
endif

nntrainer_elements = [
  'dataset',
  'layers',
  'models',
  'optimizers',
  'tensor',
  'utils',
]

foreach elem : nntrainer_elements
  subdir(elem)
  nntrainer_inc += include_directories(elem)
endforeach

nntrainer_common_sources = [
  'nntrainer_logger.cpp',
]

foreach s : nntrainer_common_sources
  nntrainer_sources += join_paths(meson.current_source_dir(), s)
endforeach

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

