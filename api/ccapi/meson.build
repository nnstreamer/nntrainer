ccapi_inc = []
ccapi_inc += include_directories('include')
ccapi_inc += include_directories('..')

# pc file is not present for 'ml-api-common' yet
if cxx.has_header('nnstreamer/ml-api-common.h', required: false)
  ccapi_inc += include_directories ('/usr/include/nnstreamer')
else
  ccapi_inc += include_directories ('../capi/include/platform')
endif

ccapi_src = []
ccapi_src += join_paths(meson.current_source_dir(), 'src', 'factory.cpp')

ccapi_headers = []
ccapi_headers += join_paths(meson.current_source_dir(), 'include', 'dataset.h')
ccapi_headers += join_paths(meson.current_source_dir(), 'include', 'layer.h')
ccapi_headers += join_paths(meson.current_source_dir(), 'include', 'model.h')
ccapi_headers += join_paths(meson.current_source_dir(), 'include', 'optimizer.h')
ccapi_headers += join_paths(meson.current_source_dir(), '..', 'nntrainer-api-common.h')

ccapi_deps = [
  nntrainer_dep
]

shared_library('ccapi-nntrainer',
  ccapi_src,
  dependencies: ccapi_deps,
  include_directories: ccapi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_ccapi_lib = static_library('ccapi-nntrainer',
  ccapi_src,
  dependencies: ccapi_deps,
  include_directories: ccapi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_ccapi_dep = declare_dependency(link_with: nntrainer_ccapi_lib,
  dependencies: ccapi_deps,
  include_directories: ccapi_inc,
)

configure_file(input: 'ccapi-nntrainer.pc.in', output: 'ccapi-nntrainer.pc',
  install_dir: join_paths(nntrainer_libdir, 'pkgconfig'),
  configuration: nntrainer_conf
)

install_headers(ccapi_headers,
  subdir: 'nntrainer'
)
