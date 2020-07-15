capi_inc = []
capi_inc += include_directories('include')
capi_inc += include_directories('..')

# pc file is not present for 'ml-api-common' yet
if cxx.has_header('nnstreamer/ml-api-common.h', required: false)
  capi_inc += include_directories ('/usr/include/nnstreamer')
else
  capi_inc += include_directories ('include/platform')
endif

capi_src = []
capi_src += join_paths(meson.current_source_dir(), 'src','nntrainer.cpp')
capi_src += join_paths(meson.current_source_dir(), 'src','nntrainer_util.cpp')

capi_headers = []
capi_headers += join_paths(meson.current_source_dir(), 'include', 'nntrainer.h')
capi_headers += join_paths(meson.current_source_dir(), '..', 'nntrainer-api-common.h')

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

configure_file(input: 'capi-nntrainer.pc.in', output: 'capi-nntrainer.pc',
  install_dir: join_paths(nntrainer_libdir, 'pkgconfig'),
  configuration: nntrainer_conf
)

install_headers( capi_headers,
  subdir: 'nntrainer'
)
