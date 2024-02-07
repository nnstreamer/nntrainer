nntrainer_inc = [
  include_directories('.'),
  include_directories('../api', '../api/ccapi/include')
]

# meson provides include_directories. But there is no support to get the absolute path from the object.
# to feed to the android.mk, sadly nntrainer_inc_abs is saved as well
nntrainer_inc_abs = [
  meson.current_source_dir(),
  meson.source_root() / 'api',
  meson.source_root() / 'api' / 'ccapi' / 'include'
]

nntrainer_sources = []
nntrainer_headers = [
   meson.current_source_dir() / 'nntrainer_log.h',
   meson.current_source_dir() / 'nntrainer_logger.h',
   meson.current_source_dir() / 'nntrainer_error.h',
   meson.current_source_dir() / 'app_context.h',
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

if get_option('enable-opencl')
  nntrainer_elements += 'opencl'
endif

foreach elem : nntrainer_elements
  subdir(elem)
  nntrainer_inc += include_directories(elem)
  nntrainer_inc_abs += meson.current_source_dir() / elem
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
  install_dir: nntrainer_includedir,
)

