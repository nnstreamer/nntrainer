capi_inc = []
capi_inc += include_directories('include')

if not get_option('enable-tizen')
  capi_inc += include_directories ('include/platform')
endif

capi_src = []
capi_src += join_paths(meson.current_source_dir(), 'src','nntrainer.cpp')

capi_headers = []
capi_headers += join_paths(meson.current_source_dir(), 'include', 'nntrainer.h')

capi_deps = [
  nntrainer_dep
]

shared_library('capi-nntrainer',
  capi_src,
  dependencies: capi_deps,
  include_directories: capi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_capi_lib = static_library('capi-nntrainer',
  capi_src,
  dependencies: capi_deps,
  include_directories: capi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_capi_dep = declare_dependency(link_with: nntrainer_capi_lib,
  dependencies: capi_deps,
  include_directories: capi_inc,
)

install_headers( capi_headers,
  subdir: 'nntrainer'
)
