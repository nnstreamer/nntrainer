nntrainer_inc = [
  include_directories('.'),
  include_directories('../api', '../api/ccapi/include')
]

nntrainer_sources = []
nntrainer_headers = [
  'nntrainer_error.h',
]

# Dependencies
nntrainer_base_deps=[
  blas_dep,
  iniparser_dep,
  ml_api_common_dep,
  libm_dep,
  libdl_dep,
  thread_dep,
  openmp_dep
]

if get_option('platform') == 'tizen'
  nntrainer_base_deps += dependency('dlog')
endif

nntrainer_elements = [
  'compiler',
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

if get_option('platform') == 'android'
  nntrainer_dep = found_dummy_dep
else
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
endif

# Install headers
install_headers(nntrainer_headers,
  subdir: 'nntrainer'
)

