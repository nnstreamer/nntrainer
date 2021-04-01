nntrainer_inc = [
  include_directories('.'),
  include_directories('../api', '../api/ccapi/include')
]

nntrainer_sources = []
nntrainer_headers = ['app_context.h', 'nntrainer_error.h', 'nntrainer_log.h', 'nntrainer_logger.h']

# Dependencies
nntrainer_base_deps=[
  blas_dep,
  iniparser_dep,
  ml_api_common_dep,
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
  'graph'
]

foreach elem : nntrainer_elements
  subdir(elem)
  nntrainer_inc += include_directories(elem)
endforeach

nntrainer_common_sources = [
  'nntrainer_logger.cpp',
  'app_context.cpp'
]

foreach s : nntrainer_common_sources
  nntrainer_sources += meson.current_source_dir() / s
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

